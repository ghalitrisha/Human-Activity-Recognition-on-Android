package com.example.har_app

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import kotlin.math.round

data class ActivityProb(val label: String, val prob: Float, val isTop: Boolean)

class ProbabilityAdapter : RecyclerView.Adapter<ProbabilityAdapter.Holder>() {

    private val items = mutableListOf<ActivityProb>()

    fun submit(list: List<ActivityProb>) {
        items.clear()
        items.addAll(list)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_activity_prob, parent, false)
        return Holder(v)
    }

    override fun onBindViewHolder(h: Holder, pos: Int) {
        val it = items[pos]
        h.tvLabel.text = it.label
        h.tvProb.text = "%.2f".format(it.prob)

        // Highlight the top class
        if (it.isTop) {
            h.itemView.setBackgroundColor(Color.parseColor("#4FC3F7")) // light blue
            h.tvLabel.setTextColor(Color.WHITE)
            h.tvProb.setTextColor(Color.WHITE)
        } else {
            h.itemView.setBackgroundColor(Color.parseColor("#F5F5F5"))
            h.tvLabel.setTextColor(Color.parseColor("#444444"))
            h.tvProb.setTextColor(Color.parseColor("#444444"))
        }
    }

    override fun getItemCount(): Int = items.size

    class Holder(v: View) : RecyclerView.ViewHolder(v) {
        val tvLabel: TextView = v.findViewById(R.id.tvLabel)
        val tvProb: TextView  = v.findViewById(R.id.tvProb)
    }
}
