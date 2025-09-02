<template>
  <div
    v-if="isVisible"
    class="flex items-center justify-center"
    :class="containerClass"
    role="status"
    :aria-label="ariaLabel"
  >
    <div
      class="animate-spin rounded-full border-solid border-t-transparent"
      :class="spinnerClass"
    ></div>
    <span v-if="showText" class="ml-3 text-sm font-medium" :class="textClass">
      {{ text }}
    </span>
    <span v-if="!showText" class="sr-only">{{ ariaLabel }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// ==== Props ====

interface Props {
  /** Whether to show the loading spinner */
  isVisible?: boolean
  /** Size of the spinner */
  size?: 'sm' | 'md' | 'lg' | 'xl'
  /** Color variant */
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'error'
  /** Loading text to display */
  text?: string
  /** Whether to show the text */
  showText?: boolean
  /** Center the spinner in its container */
  centered?: boolean
  /** Additional CSS classes for the container */
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  isVisible: true,
  size: 'md',
  variant: 'primary',
  text: 'Loading...',
  showText: true,
  centered: false,
  class: '',
})

// ==== Computed ====

const ariaLabel = computed(() => props.text || 'Loading')

const containerClass = computed(() => {
  const classes = [props.class]

  if (props.centered) {
    classes.push('min-h-24')
  }

  return classes.filter(Boolean).join(' ')
})

const spinnerClass = computed(() => {
  const classes = []

  // Size classes
  switch (props.size) {
    case 'sm':
      classes.push('h-4 w-4 border-2')
      break
    case 'md':
      classes.push('h-6 w-6 border-2')
      break
    case 'lg':
      classes.push('h-8 w-8 border-3')
      break
    case 'xl':
      classes.push('h-12 w-12 border-4')
      break
  }

  // Color classes
  switch (props.variant) {
    case 'primary':
      classes.push('border-blue-600')
      break
    case 'secondary':
      classes.push('border-gray-600')
      break
    case 'success':
      classes.push('border-green-600')
      break
    case 'warning':
      classes.push('border-yellow-600')
      break
    case 'error':
      classes.push('border-red-600')
      break
  }

  return classes.join(' ')
})

const textClass = computed(() => {
  switch (props.variant) {
    case 'primary':
      return 'text-blue-700'
    case 'secondary':
      return 'text-gray-700'
    case 'success':
      return 'text-green-700'
    case 'warning':
      return 'text-yellow-700'
    case 'error':
      return 'text-red-700'
    default:
      return 'text-gray-700'
  }
})
</script>
