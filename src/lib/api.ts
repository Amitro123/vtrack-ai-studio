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

export interface PropagateResponse {
  status: string;
  session_id: string;
  task_id: string;
  cache_path: string;
  num_frames: number;
  metadata: {
    width: number;
    height: number;
    fps: number;
    total_frames: number;
  };
  message: string;
}

export interface HealthResponse {
  status: string;
  sam3_available: boolean;
  sam3_setup_valid: boolean;
  sam3_initialized: boolean;
  device: string;
  audio_available: boolean;
  upload_dir: string;
  active_tasks: number;
}

export const api = {
  /**
   * Step 1: Pre-compute SAM3 features for video (propagation)
   * Call this before trackPointWithSession for better performance
   */
  async propagate(video: File): Promise<PropagateResponse> {
    const formData = new FormData();
    formData.append('video', video);

    const response = await fetch(`${API_BASE_URL}/api/propagate`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to propagate video');
    }

    return response.json();
  },

  /**
   * Step 2: Track point using pre-computed session (faster)
   */
  async trackPointWithSession(
    sessionId: string,
    x: number,
    y: number,
    frameIdx: number = 0,
    mode: 'fast' | 'accurate' = 'fast'
  ): Promise<TrackPointResponse> {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('x', x.toString());
    formData.append('y', y.toString());
    formData.append('frame_idx', frameIdx.toString());
    formData.append('mode', mode);

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
   * Track object from point click using SAM3 (legacy - combines propagate + track)
   */
  async trackPoint(
    video: File,
    x: number,
    y: number,
    frameIdx: number = 0,
    mode: 'fast' | 'accurate' = 'fast'
  ): Promise<TrackPointResponse> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('x', x.toString());
    formData.append('y', y.toString());
    formData.append('frame_idx', frameIdx.toString());
    formData.append('mode', mode);

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
   * Process video with text prompt using SAM3 + Demucs
   */
  async textToVideo(
    video: File,
    prompt: string,
    mode: 'fast' | 'accurate' = 'fast'
  ): Promise<TextToVideoResponse> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('prompt', prompt);
    formData.append('mode', mode);

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
   * Remove object using SAM3 + inpainting
   */
  async removeObject(
    video: File,
    x: number,
    y: number,
    frameIdx: number = 0,
    mode: 'fast' | 'accurate' = 'fast'
  ): Promise<RemoveObjectResponse> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('x', x.toString());
    formData.append('y', y.toString());
    formData.append('frame_idx', frameIdx.toString());
    formData.append('mode', mode);

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
   * Get health status
   */
  async getHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.json();
  },

  /**
   * Health check (legacy)
   */
  async healthCheck(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.json();
  },

  /**
   * Warm up SAM3 model (pre-initialize)
   */
  async warmup(): Promise<{ status: string; message: string; sam3_initialized: boolean }> {
    const response = await fetch(`${API_BASE_URL}/api/warmup`, {
      method: 'POST',
    });
    return response.json();
  },
};
