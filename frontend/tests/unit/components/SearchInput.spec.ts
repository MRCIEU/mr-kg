import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import SearchInput from '@/components/common/SearchInput.vue'

describe('SearchInput', () => {
  it('renders with placeholder text', () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: '',
        placeholder: 'Search traits...',
      },
    })

    const input = wrapper.find('input')
    expect(input.exists()).toBe(true)
    expect(input.attributes('placeholder')).toBe('Search traits...')
  })

  it('displays the current value', () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'height',
        placeholder: 'Search...',
      },
    })

    const input = wrapper.find('input')
    expect(input.element.value).toBe('height')
  })

  it('emits update:modelValue when input changes', async () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: '',
        placeholder: 'Search...',
      },
    })

    const input = wrapper.find('input')
    await input.setValue('test query')

    expect(wrapper.emitted('update:modelValue')).toBeTruthy()
    expect(wrapper.emitted('update:modelValue')![0]).toEqual(['test query'])
  })

  it('emits search event after debounce delay', async () => {
    vi.useFakeTimers()
    
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: '',
        placeholder: 'Search...',
        debounceDelay: 300,
      },
    })

    const input = wrapper.find('input')
    await input.setValue('test')

    // Fast-forward time past debounce delay
    vi.advanceTimersByTime(300)
    await nextTick()

    expect(wrapper.emitted('search')).toBeTruthy()
    expect(wrapper.emitted('search')![0]).toEqual(['test'])
    
    vi.useRealTimers()
  })

  it('shows loading spinner when searching', () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'test',
        isSearching: true,
      },
    })

    const loadingSpinner = wrapper.findComponent({ name: 'LoadingSpinner' })
    expect(loadingSpinner.exists()).toBe(true)
  })

  it('shows clear button when there is text', () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'test',
        showClearButton: true,
      },
    })

    const clearButton = wrapper.find('button[aria-label="Clear search"]')
    expect(clearButton.exists()).toBe(true)
  })

  it('clears input and emits clear event when clear button is clicked', async () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'test',
        showClearButton: true,
      },
    })

    const clearButton = wrapper.find('button[aria-label="Clear search"]')
    await clearButton.trigger('click')

    expect(wrapper.emitted('clear')).toBeTruthy()
    expect(wrapper.emitted('update:modelValue')).toBeTruthy()
    const updateEvents = wrapper.emitted('update:modelValue')!
    expect(updateEvents[updateEvents.length - 1]).toEqual([''])
  })

  it('emits search on enter key', async () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'test query',
      },
    })

    const input = wrapper.find('input')
    await input.trigger('keydown.enter')

    expect(wrapper.emitted('search')).toBeTruthy()
    expect(wrapper.emitted('search')![0]).toEqual(['test query'])
  })

  it('displays suggestions when provided', async () => {
    const suggestions = ['height', 'weight', 'BMI']
    
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'he',
        showSuggestions: true,
        suggestions,
      },
    })

    // Simulate focus to show suggestions
    const input = wrapper.find('input')
    await input.trigger('focus')

    const suggestionButtons = wrapper.findAll('button[type="button"]').filter(button => 
      !button.attributes('aria-label')?.includes('Clear')
    )
    expect(suggestionButtons).toHaveLength(suggestions.length)
  })

  it('emits suggestion-select when suggestion is clicked', async () => {
    const suggestions = ['height', 'weight', 'BMI']
    
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: 'he',
        showSuggestions: true,
        suggestions,
      },
    })

    const input = wrapper.find('input')
    await input.trigger('focus')

    const firstSuggestion = wrapper.findAll('button[type="button"]').filter(button => 
      !button.attributes('aria-label')?.includes('Clear')
    )[0]
    
    await firstSuggestion.trigger('click')

    expect(wrapper.emitted('suggestion-select')).toBeTruthy()
    expect(wrapper.emitted('suggestion-select')![0]).toEqual(['height'])
  })

  it('applies error styling when hasError is true', () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: '',
        hasError: true,
      },
    })

    const input = wrapper.find('input')
    expect(input.classes()).toContain('border-red-300')
    expect(input.classes()).toContain('focus:ring-red-500')
    expect(input.classes()).toContain('focus:border-red-500')
  })

  it('applies disabled styling when disabled', () => {
    const wrapper = mount(SearchInput, {
      props: {
        modelValue: '',
        disabled: true,
      },
    })

    const input = wrapper.find('input')
    expect(input.attributes('disabled')).toBeDefined()
    expect(input.classes()).toContain('bg-gray-50')
    expect(input.classes()).toContain('text-gray-500')
    expect(input.classes()).toContain('cursor-not-allowed')
  })
})