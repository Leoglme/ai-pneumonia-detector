<template>
  <div class="max-w-6xl mx-auto p-6 space-y-6">
    <header class="space-y-2">
      <h1 class="text-2xl font-bold">Détecteur de pneumonie (CNN & KNN)</h1>
      <p class="text-sm text-gray-600">Upload une radio pulmonaire ou clique sur un exemple pour tester.</p>
    </header>

    <div class="grid md:grid-cols-2 gap-6">
      <!-- Left column: Upload + Preview + button -->
      <div class="space-y-4">
        <UploadDropzone @file-selected="onFileSelected"/>

        <div class="rounded-xl border bg-white p-3">
          <p class="text-sm font-medium mb-2">Aperçu</p>
          <img
              v-if="!imageSamplePending"
              :src="previewUrl"
              alt="preview"
              class="w-full max-h-[400px] object-contain rounded"
          />

          <div
              v-else
              class="flex items-center justify-center h-64"
          >
            <div
                class="animate-spin rounded-full h-12 w-12 border-4 border-gray-200 border-t-gray-900"
                aria-label="Chargement"
            />
          </div>
        </div>

        <div class="flex items-center gap-3">
          <button
              class="px-4 py-2 rounded-lg bg-gray-900 text-white disabled:opacity-50"
              :disabled="loading || imageSamplePending"
              @click="onPredict"
          >
            {{ loading ? 'Analyse en cours...' : 'Analyser' }}
          </button>

          <label class="text-sm text-gray-600">
            Seuil:
            <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                v-model.number="threshold"
                class="ml-2 w-24 rounded border px-2 py-1"
            />
          </label>
        </div>

        <AlertBanner v-if="error" :message="error"/>
        <ResultPanel v-if="cnnResult" :result="cnnResult" type="cnn"/>
        <ResultPanel v-if="knnResult" :result="knnResult" type="knn"/>
      </div>

      <!-- Right column: Samples + notebook iframe-->
      <div class="space-y-4">
        <XraySampleGallery @pick="onPickSample"/>
        <div class="rounded-xl border bg-white p-4">
          <h3 class="text-lg font-semibold mb-3">Notebook (CNN vs KNN)</h3>
          <div class="aspect-video w-full">
            <iframe
                :src="iframeSrc"
                class="w-full h-full rounded-md border"
                referrerpolicy="no-referrer"
                sandbox="allow-scripts allow-same-origin"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>


<script setup lang="ts">
import UploadDropzone from '~/components/UploadDropzone.vue'
import XraySampleGallery from '~/components/XraySampleGallery.vue'
import ResultPanel from '~/components/ResultPanel.vue'
import AlertBanner from '~/components/AlertBanner.vue'
import {PneumoniaApiService} from '~/core/services/PneumoniaApiService'
import {FileLoaderService} from '~/core/services/FileLoaderService'
import type {PredictionResponse} from '~/core/types/prediction'
import type {Ref, ComputedRef} from 'vue'

// Default image File with /public/xray/pneumonia1.jpeg
const defaultFile: File = new File(
    [new Blob([new Uint8Array()])],
    'pneumonia1.jpeg',
    {type: 'image/jpeg'}
)
const selectedFile: Ref<File> = ref(defaultFile)
const previewUrl: Ref<string> = ref('/xray/pneumonia1.jpeg')
const loading: Ref<boolean> = ref(false)
const imageSamplePending: Ref<boolean> = ref(false)
const error: Ref<string | null> = ref(null)
const knnResult: Ref<PredictionResponse | null> = ref(null)
const cnnResult: Ref<PredictionResponse | null> = ref(null)
const threshold: Ref<number> = ref(0.5)

function onFileSelected(file: File, url: string) {
  selectedFile.value = file
  previewUrl.value = url
  error.value = null
  knnResult.value = null
  cnnResult.value = null
}

async function onPickSample(url: string) {
  try {
    imageSamplePending.value = true
    const file = await FileLoaderService.urlToFile(url, url.split('/').pop() || 'sample.jpg')
    const objUrl = URL.createObjectURL(file)
    onFileSelected(file, objUrl)
  } catch (e: any) {
    error.value = e?.message ?? 'Impossible de charger l’exemple.'
  } finally {
    imageSamplePending.value = false
  }
}

async function onPredictWithKnn() {
  if (!selectedFile.value) {
    error.value = 'Aucune image sélectionnée.'
    return
  }
  knnResult.value = null
  try {
    const r: PredictionResponse = await PneumoniaApiService.predictWithKnn(selectedFile.value, threshold.value)
    knnResult.value = r
  } catch (e: any) {
    // $fetch lève en cas de status code >= 400
    const msg: string = e?.data?.detail || e?.message || 'Erreur inconnue'
    error.value = `Échec de la prédiction KNN: ${msg}`
  } finally {
    loading.value = false
  }
}

async function onPredictWithCnn() {
  if (!selectedFile.value) {
    error.value = 'Aucune image sélectionnée.'
    return
  }
  cnnResult.value = null
  try {
    const r: PredictionResponse = await PneumoniaApiService.predictWithCnn(selectedFile.value, threshold.value)
    cnnResult.value = r
  } catch (e: any) {
    // $fetch lève en cas de status code >= 400
    const msg: string = e?.data?.detail || e?.message || 'Erreur inconnue'
    error.value = `Échec de la prédiction CNN: ${msg}`
  } finally {
    loading.value = false
  }
}

async function onPredict() {
  if (!selectedFile.value) {
    error.value = 'Aucune image sélectionnée.'
    return
  }
  loading.value = true
  error.value = null
  await onPredictWithKnn()
  await onPredictWithCnn()
}

const iframeSrc: ComputedRef<string> = computed(() => PneumoniaApiService.getEvaluationIframeUrl())
</script>
