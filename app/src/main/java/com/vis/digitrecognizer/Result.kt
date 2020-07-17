package com.vis.digitrecognizer
//each probablitiy of img comes here
class Result(probs: FloatArray, timeCost: Long) {
    val number: Int
    val probability: Float
    val timeCost: Long

    companion object {
        private fun argmax(probs: FloatArray): Int {
            var maxIdx = -1
            var maxProb = 0.0f
            //0-9 indices
            //higher prob value stored and its index stored
            for (i in probs.indices) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i]
                    maxIdx = i
                }
            }
            return maxIdx
        }
    }

    init {
        //final probability and the prediction is printed
        number = argmax(probs)
        probability = probs[number]
        this.timeCost = timeCost
    }
}