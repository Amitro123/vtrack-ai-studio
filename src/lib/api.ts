/**
 * VTrackAI Studio API Client
 * Handles communication with the FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ProcessingStep {
  id: string;
  label: string;
  status: 'pending' | 'processing' | 'complete' | 'error';
}

export interface TrackPointResponse {
  task_id: string;
  masked_video_url: string;
  processing_steps: ProcessingStep[];
}

export interface TextToVideoResponse {
  task_id: string;
  highlighted_video_url: string;
  audio_url: string;
  ai_response: string;
  processing_steps: ProcessingStep[];
}

export interface RemoveObjectResponse {
  task_id: string;
  inpainted_video_url: string;
  processing_steps: ProcessingStep[];
}

export const api = {
  /**
   * Track object from point click using SAM2
   */
  async trackPoint(
    video: File,
    x: number,
    y: number,
    frameIdx: number = 0
  ): Promise<TrackPointResponse> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('x', x.toString());
    formData.append('y', y.toString());
    formData.append('frame_idx', frameIdx.toString());

    const response = await fetch(`${API_BASE_URL}/api/track-point`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to track point');
    }

    return response.json();
  },

  /**
   * Process video with text prompt using GroundingDINO + SAM2 + Demucs
   */
  async textToVideo(
    video: File,
    prompt: string
  ): Promise<TextToVideoResponse> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('prompt', prompt);

    const response = await fetch(`${API_BASE_URL}/api/text-to-video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to process text-to-video');
    }

    return response.json();
  },

  /**
   * Remove object using SAM2 + ProPainter
   */
  async removeObject(
    video: File,
    x: number,
    y: number,
    frameIdx: number = 0
  ): Promise<RemoveObjectResponse> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('x', x.toString());
    formData.append('y', y.toString());
    formData.append('frame_idx', frameIdx.toString());

    const response = await fetch(`${API_BASE_URL}/api/remove-object`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to remove object');
    }

    return response.json();
  },

  /**
   * Get task status
   */
  async getTaskStatus(taskId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/task/${taskId}`);

    if (!response.ok) {
      throw new Error('Failed to get task status');
    }

    return response.json();
  },

  /**
   * Get full URL for a file
   */
  getFileUrl(path: string): string {
    if (path.startsWith('http')) {
      return path;
    }
    return `${API_BASE_URL}${path}`;
  },

  /**
   * Health check
   */
  async healthCheck(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.json();
  },
};
