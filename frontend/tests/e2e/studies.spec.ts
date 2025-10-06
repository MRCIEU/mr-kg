import { test, expect } from '@playwright/test'

test.describe('Studies View', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('navigates to studies page from home', async ({ page }) => {
    await page.click('a[href*="/studies"]')
    
    await expect(page).toHaveURL(/.*\/studies/)
    await expect(page.locator('h1')).toContainText(/study/i)
  })

  test('displays studies page header and description', async ({ page }) => {
    await page.goto('/studies')
    
    await expect(page.locator('h1')).toBeVisible()
    await expect(page.locator('h1')).toContainText(/study/i)
    
    const description = page.locator('p.text-gray-600')
    await expect(description).toBeVisible()
    await expect(description).toContainText(/metadata|research|analysis/i)
  })

  test('displays placeholder or content section', async ({ page }) => {
    await page.goto('/studies')
    
    const placeholder = page.locator('.bg-green-50')
    const hasPlaceholder = await placeholder.isVisible()
    
    if (hasPlaceholder) {
      await expect(placeholder).toContainText(/implement/i)
    } else {
      await expect(page.locator('[data-testid="studies-content"]')).toBeVisible()
    }
  })

  test('maintains navigation structure', async ({ page }) => {
    await page.goto('/studies')
    
    await expect(page.locator('nav')).toBeVisible()
    await expect(page.locator('a[href*="/traits"]')).toBeVisible()
    await expect(page.locator('a[href*="/similarities"]')).toBeVisible()
  })

  test('can navigate to other pages from studies view', async ({ page }) => {
    await page.goto('/studies')
    
    await page.click('a[href*="/traits"]')
    await expect(page).toHaveURL(/.*\/traits/)
    
    await page.goto('/studies')
    
    await page.click('a[href*="/"]')
    await expect(page).toHaveURL(/^[^/]*\/\/?$/)
  })

  test('handles direct URL access', async ({ page }) => {
    await page.goto('/studies')
    
    await expect(page.locator('h1')).toBeVisible()
    await expect(page).not.toHaveTitle(/404|not found/i)
  })

  test('maintains page state during browser navigation', async ({ page }) => {
    await page.goto('/studies')
    await page.goto('/traits')
    await page.goBack()
    
    await expect(page).toHaveURL(/.*\/studies/)
    await expect(page.locator('h1')).toContainText(/study/i)
  })
})

test.describe('Studies View - Future Implementation Tests', () => {
  test('page structure ready for study list implementation', async ({ page }) => {
    await page.goto('/studies')
    
    const mainContent = page.locator('div').first()
    await expect(mainContent).toBeVisible()
    
    const heading = page.locator('h1')
    await expect(heading).toBeVisible()
    expect(await heading.textContent()).toBeTruthy()
  })

  test('no console errors on page load', async ({ page }) => {
    const errors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })
    
    await page.goto('/studies')
    
    expect(errors.filter(e => !e.includes('favicon'))).toHaveLength(0)
  })

  test('page loads within reasonable time', async ({ page }) => {
    const startTime = Date.now()
    await page.goto('/studies')
    await page.locator('h1').waitFor()
    const loadTime = Date.now() - startTime
    
    expect(loadTime).toBeLessThan(3000)
  })
})
