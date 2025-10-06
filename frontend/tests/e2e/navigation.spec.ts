import { test, expect } from '@playwright/test'

test.describe('Cross-Page Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('completes full navigation flow through all pages', async ({ page }) => {
    await expect(page.locator('h1')).toContainText(/mr.*kg|knowledge.*graph/i)
    
    await page.click('a[href*="/traits"]')
    await expect(page).toHaveURL(/.*\/traits/)
    await expect(page.locator('h1')).toContainText(/trait/i)
    
    await page.click('a[href*="/studies"]')
    await expect(page).toHaveURL(/.*\/studies/)
    await expect(page.locator('h1')).toContainText(/stud/i)
    
    await page.click('a[href*="/similarities"]')
    await expect(page).toHaveURL(/.*\/similarities/)
    await expect(page.locator('h1')).toContainText(/similarit/i)
    
    await page.click('a[href*="/about"]')
    await expect(page).toHaveURL(/.*\/about/)
    
    await page.click('a[href="/"]')
    await expect(page).toHaveURL(/^[^/]*\/\/?$/)
  })

  test('browser back/forward navigation works correctly', async ({ page }) => {
    await page.goto('/traits')
    await page.goto('/studies')
    await page.goto('/similarities')
    
    await page.goBack()
    await expect(page).toHaveURL(/.*\/studies/)
    
    await page.goBack()
    await expect(page).toHaveURL(/.*\/traits/)
    
    await page.goForward()
    await expect(page).toHaveURL(/.*\/studies/)
    
    await page.goForward()
    await expect(page).toHaveURL(/.*\/similarities/)
  })

  test('navigation links are present on all pages', async ({ page }) => {
    const pages = ['/', '/traits', '/studies', '/similarities', '/about']
    
    for (const pagePath of pages) {
      await page.goto(pagePath)
      
      await expect(page.locator('a[href="/"]')).toBeVisible()
      await expect(page.locator('a[href*="/traits"]')).toBeVisible()
      await expect(page.locator('a[href*="/studies"]')).toBeVisible()
      await expect(page.locator('a[href*="/similarities"]')).toBeVisible()
    }
  })

  test('maintains correct active state in navigation', async ({ page }) => {
    await page.goto('/traits')
    
    const traitsLink = page.locator('a[href*="/traits"]').first()
    const classes = await traitsLink.getAttribute('class')
    
    expect(classes).toMatch(/active|current|selected|bg-|text-blue|text-indigo/)
  })

  test('page refreshes maintain correct route', async ({ page }) => {
    await page.goto('/traits')
    await page.reload()
    await expect(page).toHaveURL(/.*\/traits/)
    
    await page.goto('/studies')
    await page.reload()
    await expect(page).toHaveURL(/.*\/studies/)
    
    await page.goto('/similarities')
    await page.reload()
    await expect(page).toHaveURL(/.*\/similarities/)
  })
})

test.describe('Cross-Page Workflow: Trait to Studies', () => {
  test('navigates from trait list to studies page', async ({ page }) => {
    await page.goto('/traits')
    
    await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
    
    await page.click('a[href*="/studies"]')
    await expect(page).toHaveURL(/.*\/studies/)
  })

  test('can return to traits from studies', async ({ page }) => {
    await page.goto('/traits')
    await page.goto('/studies')
    
    await page.click('a[href*="/traits"]')
    
    await expect(page).toHaveURL(/.*\/traits/)
    await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
  })
})

test.describe('Cross-Page Workflow: Trait to Similarities', () => {
  test('navigates from trait details to similarities page', async ({ page }) => {
    await page.goto('/traits')
    
    const firstTrait = page.locator('[data-testid="trait-card"]').first()
    if (await firstTrait.isVisible()) {
      await firstTrait.click()
      await expect(page).toHaveURL(/.*\/traits\/\d+/)
      
      const similarTraitsSection = page.locator('[data-testid="similar-traits-section"]')
      if (await similarTraitsSection.isVisible()) {
        await page.click('a[href*="/similarities"]')
        await expect(page).toHaveURL(/.*\/similarities/)
      }
    }
  })

  test('can access similarities directly and navigate to traits', async ({ page }) => {
    await page.goto('/similarities')
    
    await page.click('a[href*="/traits"]')
    await expect(page).toHaveURL(/.*\/traits/)
    await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
  })
})

test.describe('Cross-Page Workflow: Studies to Traits', () => {
  test('navigates from studies to trait list', async ({ page }) => {
    await page.goto('/studies')
    
    await page.click('a[href*="/traits"]')
    
    await expect(page).toHaveURL(/.*\/traits/)
    await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
  })
})

test.describe('Filter State Preservation', () => {
  test('trait filters persist during navigation away and back', async ({ page }) => {
    await page.goto('/traits')
    
    const searchInput = page.locator('input[placeholder*="search" i]')
    if (await searchInput.isVisible()) {
      await searchInput.fill('height')
      await searchInput.press('Enter')
      
      await page.waitForURL(/.*q=height.*/)
      
      await page.click('a[href*="/studies"]')
      await expect(page).toHaveURL(/.*\/studies/)
      
      await page.goBack()
      
      await expect(page).toHaveURL(/.*q=height.*/)
    }
  })

  test('pagination state persists during navigation away and back', async ({ page }) => {
    await page.goto('/traits')
    
    const nextButton = page.locator('[data-testid="pagination-next"]')
    if (await nextButton.isVisible() && !(await nextButton.isDisabled())) {
      await nextButton.click()
      await page.waitForURL(/.*page=2.*/)
      
      await page.click('a[href*="/studies"]')
      await expect(page).toHaveURL(/.*\/studies/)
      
      await page.goBack()
      
      await expect(page).toHaveURL(/.*page=2.*/)
    }
  })
})

test.describe('Error Handling Across Pages', () => {
  test('404 navigation does not break subsequent navigation', async ({ page }) => {
    await page.goto('/nonexistent-page')
    
    await page.click('a[href="/"]')
    await expect(page).toHaveURL(/^[^/]*\/\/?$/)
    
    await page.click('a[href*="/traits"]')
    await expect(page).toHaveURL(/.*\/traits/)
  })

  test('invalid trait detail does not break navigation', async ({ page }) => {
    await page.goto('/traits/999999')
    
    await page.click('a[href*="/traits"]')
    await expect(page).toHaveURL(/.*\/traits/)
    await expect(page.locator('[data-testid="trait-card"]').first()).toBeVisible()
  })
})

test.describe('Performance: Cross-Page Navigation', () => {
  test('quick successive navigation does not cause errors', async ({ page }) => {
    const errors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })
    
    await page.goto('/')
    await page.click('a[href*="/traits"]')
    await page.click('a[href*="/studies"]')
    await page.click('a[href*="/similarities"]')
    await page.click('a[href="/"]')
    
    await page.waitForTimeout(500)
    
    expect(errors.filter(e => !e.includes('favicon'))).toHaveLength(0)
  })

  test('all pages load within reasonable time', async ({ page }) => {
    const pages = [
      { path: '/', name: 'Home' },
      { path: '/traits', name: 'Traits' },
      { path: '/studies', name: 'Studies' },
      { path: '/similarities', name: 'Similarities' },
      { path: '/about', name: 'About' }
    ]
    
    for (const { path, name } of pages) {
      const startTime = Date.now()
      await page.goto(path)
      await page.locator('h1').waitFor()
      const loadTime = Date.now() - startTime
      
      expect(loadTime, `${name} page should load quickly`).toBeLessThan(3000)
    }
  })
})
