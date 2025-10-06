import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import PaginationControls from '@/components/common/PaginationControls.vue'

describe('PaginationControls Component', () => {
  const defaultProps = {
    currentPage: 1,
    totalPages: 10,
    totalCount: 100,
    pageSize: 10,
    hasNext: true,
    hasPrevious: false
  }

  it('renders pagination information correctly', () => {
    const wrapper = mount(PaginationControls, {
      props: defaultProps
    })

    expect(wrapper.text()).toContain('Showing')
    expect(wrapper.text()).toContain('1')
    expect(wrapper.text()).toContain('10')
    expect(wrapper.text()).toContain('100')
  })

  it('disables previous button on first page', () => {
    const wrapper = mount(PaginationControls, {
      props: defaultProps
    })

    const prevButtons = wrapper.findAll('button').filter(btn => 
      btn.text().includes('Previous')
    )
    expect(prevButtons[0].attributes('disabled')).toBeDefined()
  })

  it('enables next button when hasNext is true', () => {
    const wrapper = mount(PaginationControls, {
      props: defaultProps
    })

    const nextButtons = wrapper.findAll('button').filter(btn => 
      btn.text().includes('Next')
    )
    expect(nextButtons[0].attributes('disabled')).toBeUndefined()
  })

  it('disables next button on last page', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        currentPage: 10,
        hasNext: false,
        hasPrevious: true
      }
    })

    const nextButtons = wrapper.findAll('button').filter(btn => 
      btn.text().includes('Next')
    )
    expect(nextButtons[0].attributes('disabled')).toBeDefined()
  })

  it('emits page-change event when clicking next', async () => {
    const wrapper = mount(PaginationControls, {
      props: defaultProps
    })

    const nextButtons = wrapper.findAll('button').filter(btn => 
      btn.text().includes('Next')
    )
    await nextButtons[0].trigger('click')

    expect(wrapper.emitted('page-change')).toBeTruthy()
    expect(wrapper.emitted('page-change')?.[0]).toEqual([2])
  })

  it('emits page-change event when clicking previous', async () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        currentPage: 2,
        hasPrevious: true
      }
    })

    const prevButtons = wrapper.findAll('button').filter(btn => 
      btn.text().includes('Previous')
    )
    await prevButtons[0].trigger('click')

    expect(wrapper.emitted('page-change')).toBeTruthy()
    expect(wrapper.emitted('page-change')?.[0]).toEqual([1])
  })

  it('emits page-change event when clicking page number', async () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        currentPage: 1
      }
    })

    const pageButtons = wrapper.findAll('button').filter(btn => 
      btn.text() === '3'
    )

    if (pageButtons.length > 0) {
      await pageButtons[0].trigger('click')
      expect(wrapper.emitted('page-change')?.[0]).toEqual([3])
    }
  })

  it('calculates correct start item for page 1', () => {
    const wrapper = mount(PaginationControls, {
      props: defaultProps
    })

    expect(wrapper.text()).toContain('1')
  })

  it('calculates correct start item for page 2', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        currentPage: 2,
        hasPrevious: true
      }
    })

    expect(wrapper.text()).toContain('11')
  })

  it('calculates correct end item for middle page', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        currentPage: 2,
        hasPrevious: true
      }
    })

    expect(wrapper.text()).toContain('20')
  })

  it('calculates correct end item for last partial page', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 11,
        totalPages: 11,
        totalCount: 105,
        pageSize: 10,
        hasNext: false,
        hasPrevious: true
      }
    })

    expect(wrapper.text()).toContain('105')
  })

  it('shows ellipsis for large page counts', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 5,
        totalPages: 20,
        totalCount: 200,
        pageSize: 10,
        hasNext: true,
        hasPrevious: true,
        maxVisiblePages: 7
      }
    })

    expect(wrapper.text()).toContain('...')
  })

  it('shows all pages when total pages is small', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 2,
        totalPages: 5,
        totalCount: 50,
        pageSize: 10,
        hasNext: true,
        hasPrevious: true
      }
    })

    expect(wrapper.text()).toContain('1')
    expect(wrapper.text()).toContain('2')
    expect(wrapper.text()).toContain('3')
    expect(wrapper.text()).toContain('4')
    expect(wrapper.text()).toContain('5')
  })

  it('highlights current page', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        currentPage: 3,
        hasPrevious: true
      }
    })

    const buttons = wrapper.findAll('button')
    const currentPageButton = buttons.find(btn => 
      btn.text() === '3' && 
      btn.classes().includes('bg-blue-600')
    )

    expect(currentPageButton).toBeDefined()
  })

  it('renders page size selector when enabled', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        showPageSizeSelector: true
      }
    })

    const select = wrapper.find('#page-size')
    expect(select.exists()).toBe(true)
  })

  it('hides page size selector when disabled', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        showPageSizeSelector: false
      }
    })

    const select = wrapper.find('#page-size')
    expect(select.exists()).toBe(false)
  })

  it('emits page-size-change event when changing page size', async () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        showPageSizeSelector: true
      }
    })

    const select = wrapper.find('#page-size')
    await select.setValue(25)

    expect(wrapper.emitted('page-size-change')).toBeTruthy()
    expect(wrapper.emitted('page-size-change')?.[0]).toEqual([25])
  })

  it('displays custom page size options', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        ...defaultProps,
        showPageSizeSelector: true,
        pageSizeOptions: [5, 15, 30]
      }
    })

    const select = wrapper.find('#page-size')
    const options = select.findAll('option')
    
    expect(options).toHaveLength(3)
    expect(options[0].attributes('value')).toBe('5')
    expect(options[1].attributes('value')).toBe('15')
    expect(options[2].attributes('value')).toBe('30')
  })

  it('handles zero total count correctly', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 1,
        totalPages: 0,
        totalCount: 0,
        pageSize: 10,
        hasNext: false,
        hasPrevious: false
      }
    })

    expect(wrapper.text()).toContain('0')
    expect(wrapper.text()).toContain('0')
  })

  it('renders mobile pagination controls', () => {
    const wrapper = mount(PaginationControls, {
      props: defaultProps
    })

    const mobileContainer = wrapper.find('.sm\\:hidden')
    expect(mobileContainer.exists()).toBe(true)

    const mobilePrev = mobileContainer.findAll('button').filter(btn =>
      btn.text().includes('Previous')
    )

    expect(mobilePrev.length).toBeGreaterThan(0)
  })

  it('does not emit page-change for invalid page clicks', async () => {
    const wrapper = mount(PaginationControls, {
      props: defaultProps
    })

    const currentPageButton = wrapper.findAll('button').find(btn =>
      btn.text() === '1' &&
      btn.classes().includes('bg-blue-600')
    )

    if (currentPageButton) {
      await currentPageButton.trigger('click')
      expect(wrapper.emitted('page-change')).toBeFalsy()
    }
  })
})

describe('PaginationControls Edge Cases', () => {
  it('handles single page correctly', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 1,
        totalPages: 1,
        totalCount: 5,
        pageSize: 10,
        hasNext: false,
        hasPrevious: false
      }
    })

    const prevButtons = wrapper.findAll('button').filter(btn =>
      btn.text().includes('Previous')
    )
    const nextButtons = wrapper.findAll('button').filter(btn =>
      btn.text().includes('Next')
    )

    expect(prevButtons[0].attributes('disabled')).toBeDefined()
    expect(nextButtons[0].attributes('disabled')).toBeDefined()
  })

  it('handles maximum visible pages configuration', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 10,
        totalPages: 20,
        totalCount: 200,
        pageSize: 10,
        hasNext: true,
        hasPrevious: true,
        maxVisiblePages: 5
      }
    })

    const pageButtons = wrapper.findAll('button').filter(btn =>
      /^\d+$/.test(btn.text())
    )

    expect(pageButtons.length).toBeLessThanOrEqual(5)
  })

  it('correctly positions ellipsis near start', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 2,
        totalPages: 20,
        totalCount: 200,
        pageSize: 10,
        hasNext: true,
        hasPrevious: true,
        maxVisiblePages: 7
      }
    })

    const nav = wrapper.find('nav[aria-label="Pagination"]')
    const navText = nav.text()
    const ellipsisIndex = navText.indexOf('...')
    const lastPageIndex = navText.indexOf('20')

    expect(ellipsisIndex).toBeGreaterThan(-1)
    expect(lastPageIndex).toBeGreaterThan(-1)
    expect(ellipsisIndex).toBeLessThan(lastPageIndex)
  })

  it('correctly positions ellipsis near end', () => {
    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 19,
        totalPages: 20,
        totalCount: 200,
        pageSize: 10,
        hasNext: true,
        hasPrevious: true,
        maxVisiblePages: 7
      }
    })

    const text = wrapper.text()
    const ellipsisIndex = text.indexOf('...')
    const firstPageIndex = text.indexOf('1')

    expect(ellipsisIndex).toBeGreaterThan(firstPageIndex)
  })
})
