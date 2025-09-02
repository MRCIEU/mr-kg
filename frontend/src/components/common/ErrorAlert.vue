<template>
  <div
    v-if="isVisible && (error || message)"
    class="rounded-md border p-4"
    :class="alertClass"
    role="alert"
  >
    <div class="flex">
      <div class="flex-shrink-0">
        <component
          :is="iconComponent"
          class="h-5 w-5"
          :class="iconClass"
          aria-hidden="true"
        />
      </div>
      <div class="ml-3 flex-1">
        <h3 v-if="title" class="text-sm font-medium" :class="titleClass">
          {{ title }}
        </h3>
        <div class="text-sm" :class="messageClass">
          <p v-if="message">{{ message }}</p>
          <p v-else-if="error">{{ error }}</p>
        </div>
        <div v-if="details" class="mt-2">
          <details>
            <summary
              class="cursor-pointer text-xs underline"
              :class="detailsClass"
            >
              Show details
            </summary>
            <pre
              class="mt-2 text-xs whitespace-pre-wrap"
              :class="detailsClass"
              >{{ details }}</pre
            >
          </details>
        </div>
      </div>
      <div v-if="dismissible" class="ml-auto pl-3">
        <div class="-mx-1.5 -my-1.5">
          <button
            type="button"
            class="inline-flex rounded-md p-1.5 focus:outline-none focus:ring-2 focus:ring-offset-2"
            :class="dismissButtonClass"
            @click="handleDismiss"
            aria-label="Dismiss"
          >
            <XMarkIcon class="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XMarkIcon,
} from '@heroicons/vue/24/outline'

// ==== Props ====

interface Props {
  /** Error message to display */
  error?: string
  /** Custom message to display */
  message?: string
  /** Alert title */
  title?: string
  /** Additional details (expandable) */
  details?: string
  /** Alert variant */
  variant?: 'error' | 'warning' | 'info' | 'success'
  /** Whether the alert is visible */
  isVisible?: boolean
  /** Whether the alert can be dismissed */
  dismissible?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'error',
  isVisible: true,
  dismissible: true,
})

// ==== Emits ====

const emit = defineEmits<{
  dismiss: []
}>()

// ==== Computed ====

const iconComponent = computed(() => {
  switch (props.variant) {
    case 'error':
      return ExclamationCircleIcon
    case 'warning':
      return ExclamationTriangleIcon
    case 'info':
      return InformationCircleIcon
    case 'success':
      return CheckCircleIcon
    default:
      return ExclamationCircleIcon
  }
})

const alertClass = computed(() => {
  switch (props.variant) {
    case 'error':
      return 'border-red-200 bg-red-50'
    case 'warning':
      return 'border-yellow-200 bg-yellow-50'
    case 'info':
      return 'border-blue-200 bg-blue-50'
    case 'success':
      return 'border-green-200 bg-green-50'
    default:
      return 'border-red-200 bg-red-50'
  }
})

const iconClass = computed(() => {
  switch (props.variant) {
    case 'error':
      return 'text-red-400'
    case 'warning':
      return 'text-yellow-400'
    case 'info':
      return 'text-blue-400'
    case 'success':
      return 'text-green-400'
    default:
      return 'text-red-400'
  }
})

const titleClass = computed(() => {
  switch (props.variant) {
    case 'error':
      return 'text-red-800'
    case 'warning':
      return 'text-yellow-800'
    case 'info':
      return 'text-blue-800'
    case 'success':
      return 'text-green-800'
    default:
      return 'text-red-800'
  }
})

const messageClass = computed(() => {
  switch (props.variant) {
    case 'error':
      return 'text-red-700'
    case 'warning':
      return 'text-yellow-700'
    case 'info':
      return 'text-blue-700'
    case 'success':
      return 'text-green-700'
    default:
      return 'text-red-700'
  }
})

const detailsClass = computed(() => {
  switch (props.variant) {
    case 'error':
      return 'text-red-600'
    case 'warning':
      return 'text-yellow-600'
    case 'info':
      return 'text-blue-600'
    case 'success':
      return 'text-green-600'
    default:
      return 'text-red-600'
  }
})

const dismissButtonClass = computed(() => {
  switch (props.variant) {
    case 'error':
      return 'bg-red-50 text-red-500 hover:bg-red-100 focus:ring-red-600 focus:ring-offset-red-50'
    case 'warning':
      return 'bg-yellow-50 text-yellow-500 hover:bg-yellow-100 focus:ring-yellow-600 focus:ring-offset-yellow-50'
    case 'info':
      return 'bg-blue-50 text-blue-500 hover:bg-blue-100 focus:ring-blue-600 focus:ring-offset-blue-50'
    case 'success':
      return 'bg-green-50 text-green-500 hover:bg-green-100 focus:ring-green-600 focus:ring-offset-green-50'
    default:
      return 'bg-red-50 text-red-500 hover:bg-red-100 focus:ring-red-600 focus:ring-offset-red-50'
  }
})

// ==== Methods ====

function handleDismiss() {
  emit('dismiss')
}
</script>
