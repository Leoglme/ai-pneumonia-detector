<template>
  <div v-if="result" class="rounded-xl border bg-white p-4 shadow-sm">
    <div class="flex items-center justify-between mb-3">
      <h3 class="text-lg font-semibold">Résultat</h3>
      <span
          class="inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold"
          :class="result.predicted_class === 'PNEUMONIA'
          ? 'bg-red-100 text-red-700'
          : 'bg-emerald-100 text-emerald-700'">
        {{ result.predicted_class }}
      </span>
    </div>

    <div class="space-y-3">
      <ProbabilityBar label="Probabilité NORMAL" :value="result.probability_normal" />
      <ProbabilityBar label="Probabilité PNEUMONIA" :value="result.probability_pneumonia" />
      <p class="text-xs text-gray-500">Seuil utilisé : {{ result.threshold }}</p>
    </div>
  </div>
</template>


<script setup lang="ts">
import ProbabilityBar from './ProbabilityBar.vue'
import type { PredictionResponse } from '~/core/types/prediction'

defineProps<{
  result: PredictionResponse | null
}>()
</script>
