<template>
  <div>
    <div class="flex justify-between text-sm mb-1">
      <span class="font-medium">{{ label }}</span>
      <span>{{ pct }}%</span>
    </div>
    <div class="w-full bg-gray-200 rounded-full h-3">
      <div class="h-3 rounded-full bg-gray-800" :style="{ width: pct + '%' }"></div>
    </div>
  </div>
</template>


<script setup lang="ts">
const props = defineProps<{
  label: string
  value: number // 0..1
}>()

const pct = computed(() => Math.round((props.value ?? 0) * 100))
</script>
