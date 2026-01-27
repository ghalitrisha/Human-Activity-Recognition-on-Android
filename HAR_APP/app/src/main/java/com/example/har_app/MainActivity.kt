package com.example.har_app

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.util.Locale
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : Activity() {

    private lateinit var helper: InferenceHelper

    private lateinit var tvStatus: TextView
    private lateinit var tvLatency: TextView
    private lateinit var rvProbs: RecyclerView
    private lateinit var btnLoad: Button
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    private lateinit var adapter: ProbabilityAdapter

    private val BENCHMARK_MODE = false // set true to run dummy-window benchmark

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus  = findViewById(R.id.tvStatus)
        tvLatency = findViewById(R.id.tvLatency)
        btnLoad   = findViewById(R.id.btnLoad)
        btnStart  = findViewById(R.id.btnStart)
        btnStop   = findViewById(R.id.btnStop)
        rvProbs   = findViewById(R.id.rvProbs)

        adapter = ProbabilityAdapter()
        rvProbs.layoutManager = LinearLayoutManager(this)
        rvProbs.adapter = adapter

        helper = InferenceHelper(this)

        btnLoad.setOnClickListener {
            val useNNAPI = false // Flex/LSTM -> keep CPU
            val err = helper.load(
                modelAsset = "dynamic_gru.tflite",
                labelsAsset = "labels.txt",
                useNNAPI = useNNAPI
            )
            tvStatus.text = if (err == null) {
                "Model loaded ✔ ${if (useNNAPI) "(NNAPI if available)" else "(CPU)"}"
            } else {
                "Load failed: $err"
            }
        }

        btnStart.setOnClickListener {
            if (!helper.isReady()) {
                tvStatus.text = "Load model first"
                return@setOnClickListener
            }
            if (BENCHMARK_MODE) {
                tvStatus.text = "Benchmark running…"
                runBenchmark(iters = 300, warmup = 20)
            } else {
                tvStatus.text = "Running (real-time)…"
                helper.startRealtime(periodMs = 20L) { ranked, latencyMs ->
                    // Update UI on each inference
                    tvLatency.text = "Latency: %.1f ms".format(latencyMs)
                    val top = ranked.first().first
                    adapter.submit(ranked.map { (lab, p) -> ActivityProb(lab, p, lab == top) })
                }
            }
        }

        btnStop.setOnClickListener {
            helper.stopRealtime()
            tvStatus.text = "Stopped"
        }
    }

    override fun onPause() {
        super.onPause()
        helper.stopRealtime()
    }

    override fun onDestroy() {
        super.onDestroy()
        helper.close()
    }

    // -------- Benchmark path (optional) --------
    private fun runBenchmark(iters: Int = 200, warmup: Int = 10) {
        val window = makeDummyWindow()
        repeat(warmup) { helper.predict(window) }

        val lat = DoubleArray(iters)
        android.os.Trace.beginSection("HAR_benchmark")
        for (i in 0 until iters) {
            android.os.Trace.beginSection("inference")
            val t0 = System.nanoTime()
            helper.predict(window)
            val t1 = System.nanoTime()
            android.os.Trace.endSection()
            lat[i] = (t1 - t0) / 1e6
        }
        android.os.Trace.endSection()

        val mean = lat.average()
        val sorted = lat.sorted()
        val p50 = sorted[iters / 2]
        val p90 = sorted[(iters * 9) / 10]

        // Silent CSV (no path in UI)
        try {
            val csv = buildString {
                appendLine("latency_ms")
                lat.forEach { appendLine(String.format(Locale.US, "%.3f", it)) }
            }
            val dir = getExternalFilesDir(android.os.Environment.DIRECTORY_DOWNLOADS)
            java.io.File(dir, "har_latency.csv").writeText(csv)
        } catch (_: Throwable) {}

        tvLatency.text = "Latency (mean): %.2f ms".format(Locale.US, mean)
        val summary = "μ=%.2f ms, p50=%.2f, p90=%.2f".format(Locale.US, mean, p50, p90)
        android.util.Log.i("HAR", "Benchmark $summary")
        android.widget.Toast.makeText(this, summary, android.widget.Toast.LENGTH_LONG).show()

        val ranked = helper.predictLabeled(window)
        val top = ranked.first().first
        adapter.submit(ranked.map { (lab, p) -> ActivityProb(lab, p, lab == top) })
    }

    /** Dummy 128x6 window for the benchmark/demo */
    private fun makeDummyWindow(): FloatArray {
        val T = 128; val C = 6
        val buf = FloatArray(T * C)
        for (t in 0 until T) {
            val ax = (0.5 * sin(2.0 * PI * t / 32.0)).toFloat()
            buf[t * C + 0] = ax
            buf[t * C + 1] = 0f
            buf[t * C + 2] = 0f
            buf[t * C + 3] = 0f
            buf[t * C + 4] = 0f
            buf[t * C + 5] = 0f
        }
        return buf
    }
}
