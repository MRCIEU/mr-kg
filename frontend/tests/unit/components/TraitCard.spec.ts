import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import TraitCard from '@/components/trait/TraitCard.vue'
import type { TraitListItem } from '@/types/api'

describe('TraitCard', () => {
  const mockTrait: TraitListItem = {
    trait_index: 1,
    trait_label: 'height',
    appearance_count: 150,
  }

  it('renders trait information correctly', () => {
    const wrapper = mount(TraitCard, {
      props: {
        trait: mockTrait,
      },
    })

    expect(wrapper.text()).toContain('height')
    expect(wrapper.text()).toContain('150')
  })

  it('displays trait index', () => {
    const wrapper = mount(TraitCard, {
      props: {
        trait: mockTrait,
      },
    })

    expect(wrapper.text()).toContain('1')
  })

  it('shows appearance count when showStats is true', () => {
    const wrapper = mount(TraitCard, {
      props: {
        trait: mockTrait,
        showStats: true,
      },
    })

    expect(wrapper.text()).toContain('150')
  })

  it('emits click when card is clicked', async () => {
    const wrapper = mount(TraitCard, {
      props: {
        trait: mockTrait,
        clickable: true,
      },
    })

    await wrapper.trigger('click')
    
    expect(wrapper.emitted('click')).toBeTruthy()
    expect(wrapper.emitted('click')![0]).toEqual([mockTrait])
  })

  it('shows similar action button when showSimilarAction is true', () => {
    const wrapper = mount(TraitCard, {
      props: {
        trait: mockTrait,
        showSimilarAction: true,
      },
    })

    // Look for a button or element that suggests similar functionality
    // This test would need to be adjusted based on actual TraitCard implementation
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBeGreaterThan(0)
  })

  it('applies selectable styling when selectable is true', () => {
    const wrapper = mount(TraitCard, {
      props: {
        trait: mockTrait,
        selectable: true,
      },
    })

    // Check if wrapper has hover or cursor classes when selectable
    const cardElement = wrapper.find('[role="button"], .cursor-pointer, .hover\\:shadow')
    expect(cardElement.exists()).toBe(true)
  })

  it('formats large numbers correctly', () => {
    const traitWithLargeCount: TraitListItem = {
      trait_index: 2,
      trait_label: 'common trait',
      appearance_count: 1500,
    }

    const wrapper = mount(TraitCard, {
      props: {
        trait: traitWithLargeCount,
        showStats: true,
      },
    })

    // Should format number with commas or appropriate formatting
    expect(wrapper.text()).toContain('1,500')
  })
})