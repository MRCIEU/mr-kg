<template>
  <div
    class="flex items-center justify-between border-t border-gray-200 bg-white px-4 py-3 sm:px-6"
  >
    <!-- Mobile pagination -->
    <div class="flex flex-1 justify-between sm:hidden">
      <button
        type="button"
        :disabled="!hasPrevious"
        :class="[
          'relative inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700',
          hasPrevious
            ? 'hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
            : 'cursor-not-allowed opacity-50',
        ]"
        @click="handlePrevious"
      >
        Previous
      </button>
      <button
        type="button"
        :disabled="!hasNext"
        :class="[
          'relative ml-3 inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700',
          hasNext
            ? 'hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
            : 'cursor-not-allowed opacity-50',
        ]"
        @click="handleNext"
      >
        Next
      </button>
    </div>

    <!-- Desktop pagination -->
    <div class="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
      <div>
        <p class="text-sm text-gray-700">
          Showing
          <span class="font-medium">{{ startItem }}</span>
          to
          <span class="font-medium">{{ endItem }}</span>
          of
          <span class="font-medium">{{ totalCount }}</span>
          results
        </p>
      </div>
      <div>
        <nav
          class="isolate inline-flex -space-x-px rounded-md shadow-sm"
          aria-label="Pagination"
        >
          <!-- Previous button -->
          <button
            type="button"
            :disabled="!hasPrevious"
            :class="[
              'relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300',
              hasPrevious
                ? 'hover:bg-gray-50 focus:z-20 focus:outline-none focus:ring-2 focus:ring-blue-500'
                : 'cursor-not-allowed opacity-50',
            ]"
            @click="handlePrevious"
          >
            <span class="sr-only">Previous</span>
            <ChevronLeftIcon class="h-5 w-5" aria-hidden="true" />
          </button>

          <!-- Page numbers -->
          <template v-for="pageNum in visiblePages" :key="pageNum">
            <button
              v-if="pageNum !== '...'"
              type="button"
              :class="[
                'relative inline-flex items-center px-4 py-2 text-sm font-semibold',
                pageNum === currentPage
                  ? 'z-10 bg-blue-600 text-white focus:z-20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600'
                  : 'text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-none focus:ring-2 focus:ring-blue-500',
              ]"
              @click="handlePageClick(pageNum as number)"
            >
              {{ pageNum }}
            </button>
            <span
              v-else
              class="relative inline-flex items-center px-4 py-2 text-sm font-semibold text-gray-700 ring-1 ring-inset ring-gray-300"
            >
              ...
            </span>
          </template>

          <!-- Next button -->
          <button
            type="button"
            :disabled="!hasNext"
            :class="[
              'relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300',
              hasNext
                ? 'hover:bg-gray-50 focus:z-20 focus:outline-none focus:ring-2 focus:ring-blue-500'
                : 'cursor-not-allowed opacity-50',
            ]"
            @click="handleNext"
          >
            <span class="sr-only">Next</span>
            <ChevronRightIcon class="h-5 w-5" aria-hidden="true" />
          </button>
        </nav>
      </div>
    </div>

    <!-- Page size selector -->
    <div v-if="showPageSizeSelector" class="mt-3 sm:mt-0 sm:ml-4">
      <label for="page-size" class="sr-only">Items per page</label>
      <select
        id="page-size"
        :value="pageSize"
        class="rounded-md border-gray-300 py-1.5 pl-3 pr-8 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        @change="handlePageSizeChange"
      >
        <option v-for="size in pageSizeOptions" :key="size" :value="size">
          {{ size }} per page
        </option>
      </select>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/vue/24/outline'

// ==== Props ====

interface Props {
  /** Current page number (1-based) */
  currentPage: number
  /** Total number of pages */
  totalPages: number
  /** Total number of items */
  totalCount: number
  /** Items per page */
  pageSize: number
  /** Whether there is a next page */
  hasNext: boolean
  /** Whether there is a previous page */
  hasPrevious: boolean
  /** Maximum number of page buttons to show */
  maxVisiblePages?: number
  /** Show page size selector */
  showPageSizeSelector?: boolean
  /** Available page size options */
  pageSizeOptions?: number[]
}

const props = withDefaults(defineProps<Props>(), {
  maxVisiblePages: 7,
  showPageSizeSelector: true,
  pageSizeOptions: () => [10, 25, 50, 100],
})

// ==== Emits ====

const emit = defineEmits<{
  'page-change': [page: number]
  'page-size-change': [pageSize: number]
}>()

// ==== Computed ====

const startItem = computed(() => {
  if (props.totalCount === 0) return 0
  return (props.currentPage - 1) * props.pageSize + 1
})

const endItem = computed(() => {
  const end = props.currentPage * props.pageSize
  return Math.min(end, props.totalCount)
})

const visiblePages = computed(() => {
  const pages: (number | string)[] = []
  const maxVisible = props.maxVisiblePages
  const total = props.totalPages
  const current = props.currentPage

  if (total <= maxVisible) {
    // Show all pages if total is small
    for (let i = 1; i <= total; i++) {
      pages.push(i)
    }
  } else {
    // Show subset with ellipsis
    const halfVisible = Math.floor(maxVisible / 2)

    if (current <= halfVisible + 1) {
      // Current page is near the beginning
      for (let i = 1; i <= maxVisible - 2; i++) {
        pages.push(i)
      }
      pages.push('...')
      pages.push(total)
    } else if (current >= total - halfVisible) {
      // Current page is near the end
      pages.push(1)
      pages.push('...')
      for (let i = total - (maxVisible - 3); i <= total; i++) {
        pages.push(i)
      }
    } else {
      // Current page is in the middle
      pages.push(1)
      pages.push('...')
      for (
        let i = current - halfVisible + 2;
        i <= current + halfVisible - 2;
        i++
      ) {
        pages.push(i)
      }
      pages.push('...')
      pages.push(total)
    }
  }

  return pages
})

// ==== Methods ====

function handlePageClick(page: number) {
  if (page !== props.currentPage && page >= 1 && page <= props.totalPages) {
    emit('page-change', page)
  }
}

function handlePrevious() {
  if (props.hasPrevious) {
    emit('page-change', props.currentPage - 1)
  }
}

function handleNext() {
  if (props.hasNext) {
    emit('page-change', props.currentPage + 1)
  }
}

function handlePageSizeChange(event: Event) {
  const target = event.target as HTMLSelectElement
  const newPageSize = parseInt(target.value)
  emit('page-size-change', newPageSize)
}
</script>
