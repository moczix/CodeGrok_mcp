/**
 * TypeScript type definitions and utilities.
 */

export interface User {
    id: number;
    name: string;
    email: string;
    createdAt: Date;
    metadata?: Record<string, unknown>;
}

export interface ApiResponse<T> {
    data: T;
    status: 'success' | 'error';
    message?: string;
    timestamp: number;
}

export type UserRole = 'admin' | 'user' | 'guest';

export class ValidationError extends Error {
    constructor(
        message: string,
        public field: string,
        public code: string
    ) {
        super(message);
        this.name = 'ValidationError';
    }
}

export function validateEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

export function createApiResponse<T>(data: T, status: 'success' | 'error' = 'success'): ApiResponse<T> {
    return {
        data,
        status,
        timestamp: Date.now()
    };
}

export async function fetchWithRetry<T>(
    url: string,
    options: RequestInit = {},
    retries: number = 3
): Promise<T> {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            if (i === retries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
    }
    throw new Error('Max retries exceeded');
}
