/**
 * Utility service for loading files from URLs and converting them into File objects.
 */
export class FileLoaderService {
    /**
     * Fetches a resource at the given URL and returns it as a File object.
     * @param url - Absolute or public path URL to the image (e.g., "/xray/normal1.jpg").
     * @param filename - Optional filename for the resulting File (defaults to basename of URL or "sample.jpg").
     * @param mimeType - Optional MIME type to assign to the File (defaults to "image/jpeg").
     * @throws Error when the resource cannot be fetched (non-2xx or network failure).
     */
    static async urlToFile(
        url: string,
        filename?: string,
        mimeType: string = 'image/jpeg'
    ): Promise<File> {
        const requestMode: RequestMode = 'cors'
        const res: Response = await fetch(url, { mode: requestMode, credentials: 'omit' })

        if (!res.ok) {
            const status: number = res.status
            const statusText: string = res.statusText || 'Unknown error'
            throw new Error(`Failed to load sample: ${status} ${statusText}`)
        }

        const blob: Blob = await res.blob()
        const inferredName: string = filename ?? (url.split('/').pop() || 'sample.jpg')
        return new File([blob], inferredName, {type: mimeType})
    }
}
