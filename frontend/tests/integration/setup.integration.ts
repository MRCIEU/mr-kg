import { beforeAll } from 'vitest'

const API_BASE_URL = process.env.VITE_API_BASE_URL || 'http://localhost:8000'

beforeAll(() => {
  console.log(`Running integration tests against: ${API_BASE_URL}`)
  console.log(
    'Note: These tests require a running backend. Set VITE_API_BASE_URL to configure endpoint.'
  )
})
