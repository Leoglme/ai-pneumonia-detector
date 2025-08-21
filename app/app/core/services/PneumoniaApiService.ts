import type { PredictionResponse } from '~/core/types/prediction'
import { ofetch } from 'ofetch'
import { useRuntimeConfig } from '#imports'
import type {RuntimeConfig} from "nuxt/schema";

/**
 * Service responsible for communicating with the Pneumonia KNN API.
 * Provides static helpers to access public runtime config and call the predicted endpoint.
 */
export class PneumoniaApiService {
    /**
     * Returns the API base URL from Nuxt public runtime configuration.
     * @example "http://localhost:8000"
     * @returns {string} The base URL of the API.
     */
    static getApiBaseUrl(): string {
        const cfg: RuntimeConfig = useRuntimeConfig()
        const apiBaseUrl: string | undefined =  cfg.public.apiBaseUrl
        if (!apiBaseUrl) {
            throw new Error('Missing or invalid apiBaseUrl in runtime config')
        }
        return apiBaseUrl
    }

    /**
     * Returns the iframe URL for the evaluation HTML page (CNN vs KNN).
     * @example "http://localhost:8000/api/cnn_vs_knn_evaluation/index.html"
     * @returns {string} The URL of the evaluation iframe.
     */
    static getEvaluationIframeUrl(): string {
        return `${this.getApiBaseUrl()}/api/cnn_vs_knn_evaluation/index.html`
    }

    /**
     * Calls the FastAPI KNN endpoint with a file payload and optional threshold.
     * @param {File} file - The chest X-ray image file to classify.
     * @param {number} threshold - Optional decision threshold in [0,1]. Default to 0.5.
     * @throws Error when the network request fails or returns a non-2xx status code.
     * @returns {Promise<PredictionResponse>} The prediction response containing probabilities and class.
     */
    static async predict(file: File, threshold: number = 0.5): Promise<PredictionResponse> {
        const form: FormData = new FormData()
        form.append('file', file)

        const url: string = `${this.getApiBaseUrl()}/api/knn/predict?threshold=${encodeURIComponent(threshold)}`

        // ofetch throws on HTTP errors; it returns typed data on success
        const response: PredictionResponse = await ofetch<PredictionResponse>(url, {
            method: 'POST',
            body: form
        })

        return response
    }
}
