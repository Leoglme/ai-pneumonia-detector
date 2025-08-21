<template>
  <div class="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-red-700">
    <p class="font-medium">Erreur</p>
    <p class="text-sm mt-1">{{ message }}</p>
  </div>
</template>


<script setup lang="ts">
defineProps<{ message: string }>()
</script>


