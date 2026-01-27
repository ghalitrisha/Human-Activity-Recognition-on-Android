package com.example.har_app

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.Looper
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Handles:
 *  - Loading a TFLite model and labels
 *  - Managing sensors (accelerometer + gyroscope)
 *  - Periodic sampling -> 128x6 window -> inference
 *  - Callbacks with (ranked probs, latencyMs) for UI
 */
class InferenceHelper(private val context: Context) : SensorEventListener {

    // ----- TFLite -----
    private var tflite: Interpreter? = null
    private var labels: List<String> = emptyList()
    private val T = 128
    private val C = 6

    // ----- Sensors -----
    private val sensorManager: SensorManager by lazy {
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    }
    private var accel: Sensor? = null
    private var gyro: Sensor? = null

    // Last observed values (updated by sensor callbacks)
    @Volatile private var lastAx = 0f
    @Volatile private var lastAy = 0f
    @Volatile private var lastAz = 0f
    @Volatile private var lastGx = 0f
    @Volatile private var lastGy = 0f
    @Volatile private var lastGz = 0f

    // Rolling ring buffers
    private val ringAx = FloatArray(T)
    private val ringAy = FloatArray(T)
    private val ringAz = FloatArray(T)
    private val ringGx = FloatArray(T)
    private val ringGy = FloatArray(T)
    private val ringGz = FloatArray(T)
    private var writeIdx = 0
    private var filled = false

    // Sampling loop
    private val mainHandler = Handler(Looper.getMainLooper())
    private var isSampling = false
    private var periodMs: Long = 20L
    private var resultCallback: ((List<Pair<String, Float>>, Double) -> Unit)? = null

    // ---- public API ----

    /**
     * Load TFLite model (CPU by default; Flex/LSTM models should avoid NNAPI)
     * Returns null if success, or an error string.
     */
    fun load(
        modelAsset: String = "dynamic_gru.tflite",
        labelsAsset: String = "labels.txt",
        useNNAPI: Boolean = false
    ): String? {
        return try {
            // Quick asset check to surface typos early
            val rootAssets = context.assets.list("")?.toSet() ?: emptySet()
            if (!rootAssets.contains(modelAsset) || !rootAssets.contains(labelsAsset)) {
                return "Assets not found. Put $modelAsset and $labelsAsset in app/src/main/assets/"
            }

            // Map the TFLite file
            val afd = context.assets.openFd(modelAsset)
            val fis = FileInputStream(afd.fileDescriptor)
            val model: MappedByteBuffer = fis.channel.map(
                FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.length
            )

            val opts = Interpreter.Options().apply {
                // Flex/Select TF ops typically run best on CPU
                setUseNNAPI(false)
                setNumThreads(4)
            }
            if (useNNAPI) {
                // Some models (no Flex ops) can use NNAPI; guarded try
                try { opts.setUseNNAPI(true) } catch (_: Throwable) {}
            }

            tflite?.close()
            tflite = Interpreter(model, opts)

            // Labels
            context.assets.open(labelsAsset).bufferedReader().use {
                labels = it.readLines().filter { s -> s.isNotBlank() }
            }
            null
        } catch (t: Throwable) {
            t.message ?: t.toString()
        }
    }

    fun isReady(): Boolean = (tflite != null && labels.isNotEmpty())

    /**
     * Start real-time sensing. Calls [onResult] with ranked (label, prob) and latencyMs.
     * Sampling is timer-based (~periodMs) and uses the freshest sensor values.
     */
    fun startRealtime(
        periodMs: Long = 20L,
        onResult: (List<Pair<String, Float>>, Double) -> Unit
    ) {
        if (!isReady()) throw IllegalStateException("Call load() before startRealtime()")
        if (isSampling) return

        this.periodMs = periodMs
        this.resultCallback = onResult
        resetBuffers()

        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyro  = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        val rateUs = (periodMs * 1000L).toInt()  // approximate matching
        accel?.let { sensorManager.registerListener(this, it, rateUs) }
        gyro ?.let { sensorManager.registerListener(this, it, rateUs) }

        isSampling = true
        mainHandler.post(sampleTick)
    }

    /** Stop real-time sensing and callbacks. */
    fun stopRealtime() {
        isSampling = false
        mainHandler.removeCallbacks(sampleTick)
        try { sensorManager.unregisterListener(this) } catch (_: Throwable) {}
    }

    fun close() {
        stopRealtime()
        try { tflite?.close() } catch (_: Throwable) {}
        tflite = null
    }

    // ---- SensorEventListener ----

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = e.values[0]; lastAy = e.values[1]; lastAz = e.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = e.values[0]; lastGy = e.values[1]; lastGz = e.values[2]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ---- internal ----

    private fun resetBuffers() {
        writeIdx = 0
        filled = false
        lastAx = 0f; lastAy = 0f; lastAz = 0f
        lastGx = 0f; lastGy = 0f; lastGz = 0f
    }

    private val sampleTick = object : Runnable {
        override fun run() {
            if (!isSampling) return

            // write newest readings to ring buffers
            ringAx[writeIdx] = lastAx
            ringAy[writeIdx] = lastAy
            ringAz[writeIdx] = lastAz
            ringGx[writeIdx] = lastGx
            ringGy[writeIdx] = lastGy
            ringGz[writeIdx] = lastGz

            writeIdx = (writeIdx + 1) % T
            if (writeIdx == 0) filled = true

            if (filled) {
                val window = buildWindow()
                val t0 = System.nanoTime()
                val ranked = predictLabeled(window)
                val t1 = System.nanoTime()
                val latencyMs = (t1 - t0) / 1e6
                resultCallback?.invoke(ranked, latencyMs)
            }

            mainHandler.postDelayed(this, periodMs)
        }
    }

    private fun buildWindow(): FloatArray {
        val out = FloatArray(T * C)
        var k = 0
        for (i in 0 until T) {
            val idx = (writeIdx + i) % T
            out[k++] = ringAx[idx]
            out[k++] = ringAy[idx]
            out[k++] = ringAz[idx]
            out[k++] = ringGx[idx]
            out[k++] = ringGy[idx]
            out[k++] = ringGz[idx]
        }
        return out
    }

    // ---- standalone inference helpers (still available) ----

    /** Top-1 (label, prob) from a flattened 128x6 window. */
    fun predict(windowFlattened: FloatArray): Pair<String, Float> {
        val probs = predictProbs(windowFlattened)
        val (k, p) = argmax(probs)
        val label = labels.getOrNull(k) ?: "unknown"
        return label to p
    }

    /** Full probability distribution (softmax). */
    fun predictProbs(windowFlattened: FloatArray): FloatArray {
        require(windowFlattened.size == T * C) { "Expected ${T*C} floats" }
        val interpreter = tflite ?: error("Interpreter not loaded")

        val input = Array(1) { Array(T) { FloatArray(C) } }
        var idx = 0
        for (t in 0 until T) {
            for (c in 0 until C) {
                input[0][t][c] = windowFlattened[idx++]
            }
        }

        val numClasses = labels.size.coerceAtLeast(1)
        val output = Array(1) { FloatArray(numClasses) }
        interpreter.run(input, output)

        return softmax(output[0])
    }

    /** Labeled probs sorted high→low. */
    fun predictLabeled(windowFlattened: FloatArray): List<Pair<String, Float>> {
        val p = predictProbs(windowFlattened)
        return labels.mapIndexed { i, s -> s to p[i] }.sortedByDescending { it.second }
    }

    // ---- math helpers ----
    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        var sum = 0f
        val e = FloatArray(logits.size)
        for (i in logits.indices) {
            e[i] = exp((logits[i] - max).toDouble()).toFloat()
            sum += e[i]
        }
        if (sum == 0f) return FloatArray(logits.size) { 0f }
        for (i in logits.indices) e[i] /= sum
        return e
    }

    private fun argmax(a: FloatArray): Pair<Int, Float> {
        var iMax = 0
        var v = a[0]
        for (i in 1 until a.size) if (a[i] > v) { v = a[i]; iMax = i }
        return iMax to v
    }
}
