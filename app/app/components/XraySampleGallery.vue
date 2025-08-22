<template>
  <div class="rounded-xl border bg-white p-4 shadow-sm">
    <h3 class="text-lg font-semibold mb-3">Exemples (cliquer pour tester)</h3>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <button
          v-for="s in samples"
          :key="s.url"
          type="button"
          class="group overflow-hidden rounded-lg border hover:shadow cursor-pointer"
          @click="emit('pick', s.url)"
          title="Utiliser cet exemple"
      >
        <img :src="s.url" alt="" class="aspect-square w-full object-cover" />
        <span class="p-2 text-center text-xs group-hover:bg-gray-100">
          {{ s.label }}
        </span>
      </button>
    </div>
  </div>
</template>


<script setup lang="ts">
const emit = defineEmits<{
  (e: 'pick', url: string): void
}>()

const samples = [
  { url: '/xray/normal1.jpeg', label: 'NORMAL' },
  { url: '/xray/normal2.jpeg', label: 'NORMAL' },
  { url: '/xray/pneumonia1.jpeg', label: 'PNEUMONIA' },
  { url: '/xray/pneumonia2.jpeg', label: 'PNEUMONIA' }
]
</script>
