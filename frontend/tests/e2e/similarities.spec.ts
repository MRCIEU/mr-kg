import { test, expect } from '@playwright/test'

test.describe('Similarities View', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('navigates to similarities page from home', async ({ page }) => {
    await page.click('a[href*="/similarities"]')
    
    await expect(page).toHaveURL(/.*\/similarities/)
    await expect(page.locator('h1')).toContainText(/similarit/i)
  })

  test('displays similarities page header and description', async ({ page }) => {
    await page.goto('/similarities')
    
    await expect(page.locator('h1')).toBeVisible()
    await expect(page.locator('h1')).toContainText(/trait.*similarit/i)
    
    const description = page.locator('p.text-gray-600')
    await expect(description).toBeVisible()
    await expect(description).toContainText(/similarity|embedding|analysis/i)
  })

  test('displays placeholder or content section', async ({ page }) => {
    await page.goto('/similarities')
    
    const placeholder = page.locator('.bg-purple-50')
    const hasPlaceholder = await placeholder.isVisible()
    
    if (hasPlaceholder) {
      await expect(placeholder).toContainText(/implement/i)
    } else {
      await expect(page.locator('[data-testid="similarities-content"]')).toBeVisible()
    }
  })

  test('maintains navigation structure', async ({ page }) => {
    await page.goto('/similarities')
    
    await expect(page.locator('nav')).toBeVisible()
    await expect(page.locator('a[href*="/traits"]')).toBeVisible()
    await expect(page.locator('a[href*="/studies"]')).toBeVisible()
  })

  test('can navigate to other pages from similarities view', async ({ page }) => {
    await page.goto('/similarities')
    
    await page.click('a[href*="/traits"]')
    await expect(page).toHaveURL(/.*\/traits/)
    
    await page.goto('/similarities')
    
    await page.click('a[href*="/"]')
    await expect(page).toHaveURL(/^[^/]*\/\/?$/)
  })

  test('handles direct URL access', async ({ page }) => {
    await page.goto('/similarities')
    
    await expect(page.locator('h1')).toBeVisible()
    await expect(page).not.toHaveTitle(/404|not found/i)
  })

  test('maintains page state during browser navigation', async ({ page }) => {
    await page.goto('/similarities')
    await page.goto('/traits')
    await page.goBack()
    
    await expect(page).toHaveURL(/.*\/similarities/)
    await expect(page.locator('h1')).toContainText(/similarit/i)
  })
})

test.describe('Similarities View - Future Implementation Tests', () => {
  test('page structure ready for similarity analysis implementation', async ({ page }) => {
    await page.goto('/similarities')
    
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
    
    await page.goto('/similarities')
    
    expect(errors.filter(e => !e.includes('favicon'))).toHaveLength(0)
  })

  test('page loads within reasonable time', async ({ page }) => {
    const startTime = Date.now()
    await page.goto('/similarities')
    await page.locator('h1').waitFor()
    const loadTime = Date.now() - startTime
    
    expect(loadTime).toBeLessThan(3000)
  })
})
