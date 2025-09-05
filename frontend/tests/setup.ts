import '@testing-library/jest-dom'
import { beforeAll, afterEach, afterAll } from 'vitest'
import { server } from './mocks/server'

// Override the API base URL for tests
Object.defineProperty(import.meta, 'env', {
  value: {
    VITE_API_BASE_URL: 'http://localhost:8000'
  }
})

// Silence console.log in API service during tests
const originalConsoleLog = console.log
const originalConsoleError = console.error

beforeAll(() => {
  // Start MSW server with less strict unhandled request handling
  server.listen({ onUnhandledRequest: 'warn' })
  
  // Silence noisy console logs during tests
  console.log = (...args: any[]) => {
    // Only silence API request logs
    if (typeof args[0] === 'string' && args[0].includes('API Request:')) {
      return
    }
    originalConsoleLog(...args)
  }
  
  console.error = (...args: any[]) => {
    // Only silence API error logs during tests
    if (typeof args[0] === 'string' && args[0].includes('API Error:')) {
      return
    }
    originalConsoleError(...args)
  }
})

afterEach(() => {
  // Reset handlers to initial state
  server.resetHandlers()
})

afterAll(() => {
  // Clean up
  server.close()
  console.log = originalConsoleLog
  console.error = originalConsoleError
})