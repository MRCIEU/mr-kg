<template>
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">Explore Traits</h1>
      <p class="text-gray-600 max-w-3xl">
        Browse and search trait labels from Mendelian Randomization studies.
        Explore trait relationships, appearance frequencies, and associated
        studies.
      </p>
    </div>

    <!-- Search and Filters -->
    <div class="bg-white border border-gray-200 rounded-lg p-6 mb-8">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Search Input -->
        <div class="lg:col-span-2">
          <label
            for="trait-search"
            class="block text-sm font-medium text-gray-700 mb-2"
          >
            Search Traits
          </label>
          <SearchInput
            id="trait-search"
            v-model="searchQuery"
            placeholder="Search trait labels..."
            :is-searching="traitsStore.isLoading"
            @search="handleSearch"
            @clear="handleClearSearch"
          />
        </div>

        <!-- Minimum Appearances Filter -->
        <div>
          <label
            for="min-appearances"
            class="block text-sm font-medium text-gray-700 mb-2"
          >
            Minimum Appearances
          </label>
          <select
            id="min-appearances"
            v-model="minAppearances"
            class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            @change="handleFilterChange"
          >
            <option :value="undefined">All traits</option>
            <option :value="2">2+ appearances</option>
            <option :value="5">5+ appearances</option>
            <option :value="10">10+ appearances</option>
            <option :value="25">25+ appearances</option>
            <option :value="50">50+ appearances</option>
            <option :value="100">100+ appearances</option>
          </select>
        </div>
      </div>

      <!-- Sort Options -->
      <div class="mt-4 flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <label class="text-sm font-medium text-gray-700">Sort by:</label>
          <select
            v-model="sortBy"
            class="rounded-md border-gray-300 text-sm focus:border-blue-500 focus:ring-blue-500"
            @change="handleSortChange"
          >
            <option value="appearance_count">Appearance Count</option>
            <option value="trait_label">Trait Label</option>
            <option value="trait_index">Trait Index</option>
          </select>
          <button
            type="button"
            class="text-sm text-blue-600 hover:text-blue-500"
            @click="toggleSortOrder"
          >
            {{ sortDesc ? 'Descending' : 'Ascending' }}
            <component
              :is="sortDesc ? ChevronDownIcon : ChevronUpIcon"
              class="inline h-4 w-4 ml-1"
            />
          </button>
        </div>

        <!-- View Options -->
        <div class="flex items-center space-x-2">
          <span class="text-sm text-gray-500">View:</span>
          <button
            type="button"
            :class="[
              'p-2 rounded-md',
              viewMode === 'grid'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-400 hover:text-gray-600',
            ]"
            @click="viewMode = 'grid'"
            title="Grid view"
          >
            <Squares2X2Icon class="h-5 w-5" />
          </button>
          <button
            type="button"
            :class="[
              'p-2 rounded-md',
              viewMode === 'list'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-400 hover:text-gray-600',
            ]"
            @click="viewMode = 'list'"
            title="List view"
          >
            <Bars3Icon class="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>

    <!-- Error State -->
    <ErrorAlert
      v-if="traitsStore.error"
      :error="traitsStore.error"
      variant="error"
      class="mb-6"
      @dismiss="traitsStore.loading.error = undefined"
    />

    <!-- Loading State -->
    <div
      v-if="traitsStore.isLoading && !traitsStore.hasTraits"
      class="text-center py-12"
    >
      <LoadingSpinner size="lg" text="Loading traits..." />
    </div>

    <!-- Results -->
    <div v-else-if="traitsStore.hasTraits">
      <!-- Results Summary -->
      <div class="flex items-center justify-between mb-6">
        <p class="text-sm text-gray-600">
          {{ traitsStore.totalCount.toLocaleString() }} traits found
          <span v-if="searchQuery" class="font-medium"
            >for "{{ searchQuery }}"</span
          >
        </p>

        <!-- Quick Stats -->
        <div v-if="overview" class="text-sm text-gray-500 space-x-4">
          <span>Total: {{ overview.total_traits.toLocaleString() }}</span>
          <span>Avg: {{ overview.average_appearances }} appearances</span>
        </div>
      </div>

      <!-- Traits Grid/List -->
      <div
        :class="
          viewMode === 'grid'
            ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6'
            : 'space-y-4'
        "
      >
        <TraitCard
          v-for="trait in traitsStore.traits"
          :key="trait.trait_index"
          :trait="trait"
          :selectable="false"
          :show-stats="true"
          :show-similar-action="true"
          @view-details="handleViewDetails"
          @find-similar="handleFindSimilar"
          @click="handleTraitClick"
        />
      </div>

      <!-- Loading overlay for pagination -->
      <div v-if="traitsStore.isLoading" class="relative">
        <div
          class="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center"
        >
          <LoadingSpinner text="Loading more traits..." />
        </div>
      </div>

      <!-- Pagination -->
      <div class="mt-8">
        <PaginationControls
          :current-page="traitsStore.currentPage"
          :total-pages="traitsStore.totalPages"
          :total-count="traitsStore.totalCount"
          :page-size="traitsStore.pageSize"
          :has-next="traitsStore.hasNext"
          :has-previous="traitsStore.hasPrevious"
          @page-change="handlePageChange"
          @page-size-change="handlePageSizeChange"
        />
      </div>
    </div>

    <!-- Empty State -->
    <div v-else class="text-center py-12">
      <div class="mx-auto max-w-md">
        <MagnifyingGlassIcon class="mx-auto h-12 w-12 text-gray-400" />
        <h3 class="mt-4 text-lg font-medium text-gray-900">No traits found</h3>
        <p class="mt-2 text-sm text-gray-500">
          <span v-if="searchQuery">
            No traits match your search for "{{ searchQuery }}".
          </span>
          <span v-else> No traits match your current filters. </span>
        </p>
        <div class="mt-6">
          <button
            type="button"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
            @click="handleClearFilters"
          >
            Clear filters
          </button>
        </div>
      </div>
    </div>

    <!-- Trait Detail Modal (placeholder) -->
    <!-- This would be implemented as a separate component -->
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  Bars3Icon,
  ChevronDownIcon,
  ChevronUpIcon,
  MagnifyingGlassIcon,
  Squares2X2Icon,
} from '@heroicons/vue/24/outline'

import { useTraitsStore } from '@/stores/traits'
import type { TraitListItem, TraitsOverview } from '@/types/api'

// Components
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import ErrorAlert from '@/components/common/ErrorAlert.vue'
import SearchInput from '@/components/common/SearchInput.vue'
import PaginationControls from '@/components/common/PaginationControls.vue'
import TraitCard from '@/components/trait/TraitCard.vue'

// ==== Setup ====

const router = useRouter()
const traitsStore = useTraitsStore()

// ==== State ====

const searchQuery = ref('')
const minAppearances = ref<number | undefined>(undefined)
const sortBy = ref('appearance_count')
const sortDesc = ref(true)
const viewMode = ref<'grid' | 'list'>('grid')
const overview = ref<TraitsOverview | null>(null)

// ==== Computed ====

// ==== Methods ====

async function loadTraits() {
  await traitsStore.fetchTraits({
    order_by: sortBy.value,
    order_desc: sortDesc.value,
    min_appearances: minAppearances.value,
  })
}

async function loadOverview() {
  if (!overview.value) {
    try {
      await traitsStore.fetchTraitsOverview()
      overview.value = traitsStore.traitsOverview
    } catch (error) {
      console.error('Failed to load traits overview:', error)
    }
  }
}

function handleSearch(query: string) {
  searchQuery.value = query
  if (query.trim()) {
    traitsStore.searchTraits(query, {
      min_appearances: minAppearances.value,
    })
  } else {
    loadTraits()
  }
}

function handleClearSearch() {
  searchQuery.value = ''
  loadTraits()
}

function handleFilterChange() {
  traitsStore.updateFilters({
    min_appearances: minAppearances.value,
  })

  if (searchQuery.value) {
    handleSearch(searchQuery.value)
  } else {
    loadTraits()
  }
}

function handleSortChange() {
  if (searchQuery.value) {
    handleSearch(searchQuery.value)
  } else {
    loadTraits()
  }
}

function toggleSortOrder() {
  sortDesc.value = !sortDesc.value
  handleSortChange()
}

function handlePageChange(page: number) {
  traitsStore.goToPage(page)
}

function handlePageSizeChange(pageSize: number) {
  traitsStore.pageSize = pageSize
  traitsStore.currentPage = 1
  if (searchQuery.value) {
    handleSearch(searchQuery.value)
  } else {
    loadTraits()
  }
}

function handleTraitClick(trait: TraitListItem) {
  router.push(`/traits/${trait.trait_index}`)
}

function handleViewDetails(trait: TraitListItem) {
  router.push(`/traits/${trait.trait_index}`)
}

function handleFindSimilar(trait: TraitListItem) {
  router.push(`/traits/${trait.trait_index}/similar`)
}

function handleClearFilters() {
  searchQuery.value = ''
  minAppearances.value = undefined
  sortBy.value = 'appearance_count'
  sortDesc.value = true
  traitsStore.clearFilters()
  loadTraits()
}

// ==== Lifecycle ====

onMounted(async () => {
  await Promise.all([loadTraits(), loadOverview()])
})

// ==== Watchers ====

watch(
  () => traitsStore.filters,
  () => {
    // Sync local state with store filters
    minAppearances.value = traitsStore.filters.min_appearances
  },
  { deep: true }
)
</script>
