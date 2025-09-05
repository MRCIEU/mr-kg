import { test, expect } from '@playwright/test'

test.describe('Traits Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('navigates to traits page and views trait list', async ({ page }) => {
    await page.click('a[href*="/traits"]')
    
    await expect(page).toHaveURL(/.*\/traits/)
    await expect(page.locator('h1')).toContainText(/traits/i)
    
    await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
  })

  test('searches for traits', async ({ page }) => {
    await page.goto('/traits')
    
    const searchInput = page.locator('input[placeholder*="search" i]')
    await searchInput.fill('height')
    await searchInput.press('Enter')
    
    await expect(page.locator('[data-testid="trait-card"]')).toContainText(/height/i)
  })

  test('views trait details', async ({ page }) => {
    await page.goto('/traits')
    
    await page.locator('[data-testid="trait-card"]').first().click()
    
    await expect(page).toHaveURL(/.*\/traits\/\d+/)
    await expect(page.locator('[data-testid="trait-label"]')).toBeVisible()
    await expect(page.locator('[data-testid="trait-statistics"]')).toBeVisible()
  })

  test('displays similar traits', async ({ page }) => {
    await page.goto('/traits/1')
    
    await expect(page.locator('[data-testid="similar-traits-section"]')).toBeVisible()
    await expect(page.locator('[data-testid="similar-trait-item"]').first()).toBeVisible()
  })

  test('views traits overview page', async ({ page }) => {
    await page.goto('/traits/overview')
    
    await expect(page.locator('h1')).toContainText(/overview/i)
    await expect(page.locator('[data-testid="overview-stats"]')).toBeVisible()
    await expect(page.locator('[data-testid="top-traits-section"]')).toBeVisible()
  })

  test('filters traits by minimum appearances', async ({ page }) => {
    await page.goto('/traits')
    
    const filterButton = page.locator('[data-testid="filter-button"]')
    if (await filterButton.isVisible()) {
      await filterButton.click()
      
      const minAppearancesInput = page.locator('input[name="min_appearances"]')
      await minAppearancesInput.fill('100')
      
      const applyButton = page.locator('[data-testid="apply-filters"]')
      await applyButton.click()
      
      await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
    }
  })

  test('navigates through trait pages', async ({ page }) => {
    await page.goto('/traits')
    
    const nextButton = page.locator('[data-testid="pagination-next"]')
    if (await nextButton.isVisible() && !(await nextButton.isDisabled())) {
      await nextButton.click()
      await expect(page).toHaveURL(/.*page=2/)
    }
    
    const prevButton = page.locator('[data-testid="pagination-prev"]')
    if (await prevButton.isVisible() && !(await prevButton.isDisabled())) {
      await prevButton.click()
      await expect(page).toHaveURL(/.*page=1/)
    }
  })

  test('shows loading states during data fetch', async ({ page }) => {
    await page.goto('/traits')
    
    await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible()
    
    await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
    await expect(page.locator('[data-testid="loading-spinner"]')).not.toBeVisible()
  })

  test('handles search with no results', async ({ page }) => {
    await page.goto('/traits')
    
    const searchInput = page.locator('input[placeholder*="search" i]')
    await searchInput.fill('nonexistenttraitname123')
    await searchInput.press('Enter')
    
    await expect(page.locator('[data-testid="no-results"]')).toBeVisible()
    await expect(page.locator('[data-testid="no-results"]')).toContainText(/no.*found/i)
  })

  test('clears search and returns to full list', async ({ page }) => {
    await page.goto('/traits')
    
    const searchInput = page.locator('input[placeholder*="search" i]')
    await searchInput.fill('height')
    await searchInput.press('Enter')
    
    const clearButton = page.locator('[data-testid="clear-search"]')
    if (await clearButton.isVisible()) {
      await clearButton.click()
    } else {
      await searchInput.clear()
      await searchInput.press('Enter')
    }
    
    await expect(page.locator('[data-testid="trait-card"]')).toHaveCount(5)
  })
})