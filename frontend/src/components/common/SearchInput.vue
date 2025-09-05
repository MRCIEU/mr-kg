<template>
  <div class="relative">
    <div class="relative">
      <div
        class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3"
      >
        <MagnifyingGlassIcon class="h-5 w-5 text-gray-400" aria-hidden="true" />
      </div>
      <input
        :id="inputId"
        ref="inputRef"
        v-model="localValue"
        type="text"
        :name="name"
        :placeholder="placeholder"
        :disabled="disabled"
        :class="inputClass"
        :aria-describedby="ariaDescribedby"
        data-testid="search-input"
        @input="handleInput"
        @focus="handleFocus"
        @blur="handleBlur"
        @keydown.enter="handleEnter"
        @keydown.escape="handleEscape"
      />
      <div
        v-if="showClearButton"
        class="absolute inset-y-0 right-0 flex items-center pr-3"
      >
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-full p-1"
          @click="handleClear"
          aria-label="Clear search"
          data-testid="clear-search"
        >
          <XMarkIcon class="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </div>

    <!-- Loading indicator -->
    <div
      v-if="isSearching"
      class="absolute inset-y-0 right-0 flex items-center pr-3"
      :class="{ 'pr-10': showClearButton }"
      data-testid="loading-spinner"
    >
      <LoadingSpinner size="sm" :show-text="false" />
    </div>

    <!-- Search suggestions dropdown -->
    <div
      v-if="showSuggestions && (suggestions.length > 0 || showNoResults)"
      class="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-md shadow-lg max-h-60 overflow-auto"
    >
      <div
        v-if="suggestions.length === 0 && showNoResults"
        class="px-3 py-2 text-sm text-gray-500"
        data-testid="no-results"
      >
        No results found
      </div>
      <button
        v-for="(suggestion, index) in suggestions"
        :key="getSuggestionKey(suggestion, index)"
        type="button"
        class="w-full px-3 py-2 text-left text-sm hover:bg-gray-50 focus:bg-gray-50 focus:outline-none"
        :class="{ 'bg-gray-50': index === highlightedIndex }"
        @click="handleSuggestionClick(suggestion)"
        @mouseenter="highlightedIndex = index"
      >
        <slot name="suggestion" :suggestion="suggestion" :index="index">
          {{ getSuggestionText(suggestion) }}
        </slot>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/vue/24/outline'
import { debouncedRef } from '@vueuse/core'
import LoadingSpinner from './LoadingSpinner.vue'

// ==== Props ====

interface Props {
  /** Current search value */
  modelValue: string
  /** Input placeholder text */
  placeholder?: string
  /** Input name attribute */
  name?: string
  /** Input ID */
  inputId?: string
  /** Whether input is disabled */
  disabled?: boolean
  /** Debounce delay in milliseconds */
  debounceDelay?: number
  /** Whether to show search suggestions */
  showSuggestions?: boolean
  /** Array of suggestions to display */
  suggestions?: any[]
  /** Whether currently searching/loading */
  isSearching?: boolean
  /** Show "no results" message when suggestions are empty */
  showNoResults?: boolean
  /** Show clear button when there's text */
  showClearButton?: boolean
  /** Size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Error state */
  hasError?: boolean
  /** Function to get text from suggestion object */
  suggestionTextKey?: string
  /** Function to get unique key from suggestion */
  suggestionKeyFn?: (suggestion: any, index: number) => string | number
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: 'Search...',
  debounceDelay: 300,
  showSuggestions: false,
  suggestions: () => [],
  isSearching: false,
  showNoResults: true,
  showClearButton: true,
  size: 'md',
  hasError: false,
  suggestionTextKey: 'label',
})

// ==== Emits ====

const emit = defineEmits<{
  'update:modelValue': [value: string]
  search: [query: string]
  'suggestion-select': [suggestion: any]
  focus: []
  blur: []
  clear: []
}>()

// ==== Refs ====

const inputRef = ref<HTMLInputElement | null>(null)
const localValue = ref(props.modelValue)
const debouncedValue = debouncedRef(localValue, props.debounceDelay)
const isFocused = ref(false)
const highlightedIndex = ref(-1)

// ==== Computed ====

const inputClass = computed(() => {
  const baseClasses = [
    'block w-full pl-10 border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500',
  ]

  // Size classes
  switch (props.size) {
    case 'sm':
      baseClasses.push('text-sm py-1.5 pr-3')
      break
    case 'md':
      baseClasses.push('text-sm py-2 pr-3')
      break
    case 'lg':
      baseClasses.push('text-base py-3 pr-4')
      break
  }

  // Clear button spacing
  if (props.showClearButton && localValue.value) {
    baseClasses.push('pr-10')
  }

  // Error state
  if (props.hasError) {
    baseClasses.push('border-red-300 focus:ring-red-500 focus:border-red-500')
  }

  // Disabled state
  if (props.disabled) {
    baseClasses.push('bg-gray-50 text-gray-500 cursor-not-allowed')
  }

  return baseClasses.join(' ')
})

const ariaDescribedby = computed(() => {
  const ids = []
  if (props.hasError) ids.push(`${props.inputId}-error`)
  return ids.length > 0 ? ids.join(' ') : undefined
})

const showSuggestions = computed(() => {
  return props.showSuggestions && isFocused.value && localValue.value.length > 0
})

// ==== Watchers ====

watch(
  () => props.modelValue,
  (newValue) => {
    localValue.value = newValue
  }
)

watch(localValue, (newValue) => {
  emit('update:modelValue', newValue)
})

watch(debouncedValue, (newValue) => {
  if (newValue !== props.modelValue) {
    emit('search', newValue)
  }
})

// ==== Methods ====

function handleInput(event: Event) {
  const target = event.target as HTMLInputElement
  localValue.value = target.value
  highlightedIndex.value = -1
}

function handleFocus() {
  isFocused.value = true
  emit('focus')
}

function handleBlur() {
  // Delay blur to allow suggestion clicks
  setTimeout(() => {
    isFocused.value = false
    highlightedIndex.value = -1
    emit('blur')
  }, 200)
}

function handleEnter(event: KeyboardEvent) {
  if (showSuggestions.value && highlightedIndex.value >= 0) {
    event.preventDefault()
    handleSuggestionClick(props.suggestions[highlightedIndex.value])
  } else {
    emit('search', localValue.value)
  }
}

function handleEscape() {
  if (showSuggestions.value) {
    isFocused.value = false
    highlightedIndex.value = -1
  } else {
    handleClear()
  }
}

function handleClear() {
  localValue.value = ''
  highlightedIndex.value = -1
  emit('clear')
  nextTick(() => {
    inputRef.value?.focus()
  })
}

function handleSuggestionClick(suggestion: any) {
  localValue.value = getSuggestionText(suggestion)
  isFocused.value = false
  highlightedIndex.value = -1
  emit('suggestion-select', suggestion)
}

function getSuggestionText(suggestion: any): string {
  if (typeof suggestion === 'string') return suggestion
  if (typeof suggestion === 'object' && suggestion[props.suggestionTextKey]) {
    return suggestion[props.suggestionTextKey]
  }
  return String(suggestion)
}

function getSuggestionKey(suggestion: any, index: number): string | number {
  if (props.suggestionKeyFn) {
    return props.suggestionKeyFn(suggestion, index)
  }
  if (typeof suggestion === 'object' && suggestion.id) {
    return suggestion.id
  }
  return index
}

// Keyboard navigation for suggestions
function handleKeydown(event: KeyboardEvent) {
  if (!showSuggestions.value) return

  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault()
      highlightedIndex.value = Math.min(
        highlightedIndex.value + 1,
        props.suggestions.length - 1
      )
      break
    case 'ArrowUp':
      event.preventDefault()
      highlightedIndex.value = Math.max(highlightedIndex.value - 1, -1)
      break
  }
}

// ==== Lifecycle ====

onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})

// ==== Public Methods ====

function focus() {
  inputRef.value?.focus()
}

function blur() {
  inputRef.value?.blur()
}

defineExpose({
  focus,
  blur,
})
</script>
