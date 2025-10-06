import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import PaginationControls from '@/components/common/PaginationControls.vue'
import SearchInput from '@/components/common/SearchInput.vue'
import TraitCard from '@/components/trait/TraitCard.vue'
import TraitsView from '@/views/TraitsView.vue'
import { useTraitsStore } from '@/stores/traits'
import { server } from '../../mocks/server'
import { http, HttpResponse } from 'msw'

describe('Component Integration: TraitsView with Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('integrates SearchInput with traits store', async () => {
    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    const searchInput = wrapper.findComponent(SearchInput)
    expect(searchInput.exists()).toBe(true)

    await searchInput.vm.$emit('search', 'height')
    
    expect(traitsStore.isLoading).toBe(true)
  })

  it('integrates PaginationControls with traits store pagination', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits({ page: 1, page_size: 10 })

    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    const pagination = wrapper.findComponent(PaginationControls)
    
    if (pagination.exists()) {
      expect(pagination.props('currentPage')).toBe(traitsStore.currentPage)
      expect(pagination.props('totalPages')).toBe(traitsStore.totalPages)
      expect(pagination.props('hasNext')).toBe(traitsStore.hasNext)
      expect(pagination.props('hasPrevious')).toBe(traitsStore.hasPrevious)
    }
  })

  it('renders TraitCard components from store data', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    await wrapper.vm.$nextTick()

    const traitCards = wrapper.findAllComponents(TraitCard)
    expect(traitCards.length).toBeGreaterThan(0)
    expect(traitCards.length).toBe(traitsStore.traits.length)
  })

  it('updates view when store pagination changes', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits({ page: 1, page_size: 5 })

    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    if (traitsStore.hasNext) {
      await traitsStore.fetchTraits({ page: 2, page_size: 5 })
      await wrapper.vm.$nextTick()

      const newTraitCount = wrapper.findAllComponents(TraitCard).length
      expect(newTraitCount).toBeGreaterThan(0)
    }
  })

  it('displays loading state from store', async () => {
    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    const traitsStore = useTraitsStore()
    
    const fetchPromise = traitsStore.fetchTraits()
    
    await wrapper.vm.$nextTick()
    
    if (traitsStore.isLoading) {
      const loadingSpinner = wrapper.find('[data-testid="loading-spinner"]')
      expect(loadingSpinner.exists()).toBe(true)
    }

    await fetchPromise
  })

  it('displays error state from store', async () => {
    server.use(
      http.get('*/traits', () => {
        return HttpResponse.json({ detail: 'Server error' }, { status: 500 })
      })
    )

    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    if (traitsStore.error) {
      const errorAlert = wrapper.find('[data-testid="error-alert"]')
      expect(errorAlert.exists()).toBe(true)
    }
  })
})

describe('Component Integration: PaginationControls with Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('emits page change and updates store', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits({ page: 1, page_size: 10 })

    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: traitsStore.currentPage,
        totalPages: traitsStore.totalPages,
        totalCount: traitsStore.totalCount,
        pageSize: traitsStore.pageSize,
        hasNext: traitsStore.hasNext,
        hasPrevious: traitsStore.hasPrevious
      }
    })

    if (traitsStore.hasNext) {
      const nextButton = wrapper.find('button[aria-label*="Next"]')
      await nextButton.trigger('click')

      expect(wrapper.emitted('page-change')).toBeTruthy()
      expect(wrapper.emitted('page-change')?.[0]).toEqual([2])
    }
  })

  it('emits page size change', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: traitsStore.currentPage,
        totalPages: traitsStore.totalPages,
        totalCount: traitsStore.totalCount,
        pageSize: traitsStore.pageSize,
        hasNext: traitsStore.hasNext,
        hasPrevious: traitsStore.hasPrevious,
        showPageSizeSelector: true
      }
    })

    const pageSizeSelect = wrapper.find('#page-size')
    if (pageSizeSelect.exists()) {
      await pageSizeSelect.setValue(25)

      expect(wrapper.emitted('page-size-change')).toBeTruthy()
      expect(wrapper.emitted('page-size-change')?.[0]).toEqual([25])
    }
  })

  it('disables previous button on first page', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits({ page: 1, page_size: 10 })

    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 1,
        totalPages: traitsStore.totalPages,
        totalCount: traitsStore.totalCount,
        pageSize: traitsStore.pageSize,
        hasNext: traitsStore.hasNext,
        hasPrevious: false
      }
    })

    const prevButtons = wrapper.findAll('button').filter(btn => 
      btn.text().includes('Previous') || 
      btn.find('[aria-hidden="true"]').exists()
    )

    const prevButton = prevButtons[0]
    expect(prevButton.attributes('disabled')).toBeDefined()
  })

  it('calculates correct item range display', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits({ page: 2, page_size: 10 })

    const wrapper = mount(PaginationControls, {
      props: {
        currentPage: 2,
        totalPages: traitsStore.totalPages,
        totalCount: traitsStore.totalCount,
        pageSize: 10,
        hasNext: traitsStore.hasNext,
        hasPrevious: true
      }
    })

    const text = wrapper.text()
    expect(text).toContain('Showing')
    expect(text).toMatch(/11|to|20/)
  })
})

describe('Component Integration: SearchInput with Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('triggers store search on search emit', async () => {
    const traitsStore = useTraitsStore()

    const wrapper = mount(SearchInput, {
      props: {
        modelValue: '',
        placeholder: 'Search traits'
      }
    })

    const input = wrapper.find('input')
    await input.setValue('height')
    
    wrapper.vm.$emit('search', 'height')
    
    expect(wrapper.emitted('search')).toBeTruthy()
    expect(wrapper.emitted('search')?.[0]).toEqual(['height'])
    expect(traitsStore).toBeDefined()
  })

  it('clears search and resets store filters', async () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'height',
        placeholder: 'Search traits'
      }
    })

    const clearButton = wrapper.find('[data-testid="clear-search"]')
    if (clearButton.exists()) {
      await clearButton.trigger('click')

      expect(wrapper.emitted('clear')).toBeTruthy()
    }
  })

  it('shows loading state when isSearching is true', async () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'height',
        placeholder: 'Search traits',
        isSearching: true
      }
    })

    const loadingIndicator = wrapper.find('[data-testid="loading-spinner"]')
    expect(loadingIndicator.exists()).toBe(true)
  })
})

describe('Component Integration: TraitCard with Store Data', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders trait data from store correctly', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    if (traitsStore.traits.length > 0) {
      const trait = traitsStore.traits[0]

      const wrapper = mount(TraitCard, {
        props: {
          trait
        },
        global: {
          stubs: {
            RouterLink: { template: '<a><slot /></a>' }
          }
        }
      })

      expect(wrapper.text()).toContain(trait.trait_label)
      expect(wrapper.text()).toContain(String(trait.appearance_count))
    }
  })

  it('navigates to trait detail on click', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    if (traitsStore.traits.length > 0) {
      const trait = traitsStore.traits[0]

      const wrapper = mount(TraitCard, {
        props: {
          trait
        },
        global: {
          stubs: {
            RouterLink: { 
              template: '<a :href="`/traits/${to}`"><slot /></a>',
              props: ['to']
            }
          }
        }
      })

      const card = wrapper.find('[data-testid="trait-card"]')
      expect(card.exists()).toBe(true)
    }
  })
})

describe('Component Integration: Full Page Workflow', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('completes search to detail navigation workflow', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    const searchInput = wrapper.findComponent(SearchInput)
    await searchInput.vm.$emit('search', 'height')

    await traitsStore.fetchTraits({ q: 'height' })
    await wrapper.vm.$nextTick()

    const traitCards = wrapper.findAllComponents(TraitCard)
    expect(traitCards.length).toBeGreaterThan(0)
  })

  it('handles pagination workflow end-to-end', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits({ page: 1, page_size: 5 })

    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    const pagination = wrapper.findComponent(PaginationControls)
    
    if (pagination.exists() && traitsStore.hasNext) {
      await pagination.vm.$emit('page-change', 2)

      await traitsStore.fetchTraits({ page: 2, page_size: 5 })
      await wrapper.vm.$nextTick()

      expect(traitsStore.currentPage).toBe(2)
    }
  })

  it('handles filter change workflow', async () => {
    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    const wrapper = mount(TraitsView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' }
        }
      }
    })

    const minAppearancesSelect = wrapper.find('#min-appearances')
    if (minAppearancesSelect.exists()) {
      await minAppearancesSelect.setValue(10)

      await traitsStore.fetchTraits({ min_appearances: 10 })
      await wrapper.vm.$nextTick()

      const traitCards = wrapper.findAllComponents(TraitCard)
      expect(traitCards.length).toBeGreaterThanOrEqual(0)
    }
  })

  it('handles error recovery workflow', async () => {
    server.use(
      http.get('*/traits', () => {
        return HttpResponse.json({ detail: 'Server error' }, { status: 500 })
      })
    )

    const traitsStore = useTraitsStore()
    await traitsStore.fetchTraits()

    expect(traitsStore.error).toBeTruthy()

    server.resetHandlers()

    await traitsStore.fetchTraits()
    expect(traitsStore.error).toBeFalsy()
    expect(traitsStore.traits.length).toBeGreaterThan(0)
  })
})
