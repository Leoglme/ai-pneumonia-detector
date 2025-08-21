<template>
  <div
      class="rounded-xl border-2 border-dashed p-6 text-center bg-white hover:bg-gray-50 transition"
      :class="isDragging ? 'border-gray-900' : 'border-gray-300'"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop="onDrop"
  >
    <p class="font-medium mb-2">Glisse ta radio ici</p>
    <p class="text-sm text-gray-500 mb-4">ou</p>
    <button type="button" class="px-4 py-2 rounded-lg bg-gray-900 text-white" @click="onBrowse">
      Choisir une image
    </button>
    <input
        ref="inputRef"
        type="file"
        accept="image/*"
        class="hidden"
        @change="onFileChange"
    />
  </div>
</template>


<script setup lang="ts">
const emit = defineEmits<{
  (e: 'file-selected', file: File, previewUrl: string): void
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)

function onBrowse() {
  inputRef.value?.click()
}

function onFileChange(e: Event) {
  const files = (e.target as HTMLInputElement).files
  if (!files || !files[0]) return
  const file = files[0]
  const url = URL.createObjectURL(file)
  emit('file-selected', file, url)
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  isDragging.value = false
  const file = e.dataTransfer?.files?.[0]
  if (!file) return
  const url = URL.createObjectURL(file)
  emit('file-selected', file, url)
}
</script>
