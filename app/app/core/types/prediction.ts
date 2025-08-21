export type PredictionResponse = {
    predicted_class: string
    probability_pneumonia: number
    probability_normal: number
    threshold: number
    metadata: Record<string, unknown>
}

export type ApiError = {
    status: number
    message: string
}
