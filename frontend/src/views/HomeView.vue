<template>
  <div class="max-w-7xl mx-auto">
    <!-- Hero Section -->
    <div class="text-center mb-12">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">MR-KG Explorer</h1>
      <p class="text-xl text-gray-600 max-w-3xl mx-auto">
        Explore Mendelian Randomization Knowledge Graph - Browse trait labels,
        study details, and similarity relationships extracted from PubMed
        literature using large language models.
      </p>
    </div>

    <!-- Quick Stats -->
    <div v-if="overview" class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
      <div class="bg-white border border-gray-200 rounded-lg p-6 text-center">
        <div class="text-3xl font-bold text-blue-600 mb-2">
          {{ overview.total_traits.toLocaleString() }}
        </div>
        <div class="text-sm text-gray-600">Total Traits</div>
      </div>

      <div class="bg-white border border-gray-200 rounded-lg p-6 text-center">
        <div class="text-3xl font-bold text-green-600 mb-2">
          {{ overview.total_appearances.toLocaleString() }}
        </div>
        <div class="text-sm text-gray-600">Total Appearances</div>
      </div>

      <div class="bg-white border border-gray-200 rounded-lg p-6 text-center">
        <div class="text-3xl font-bold text-purple-600 mb-2">
          {{ overview.average_appearances }}
        </div>
        <div class="text-sm text-gray-600">Average Appearances</div>
      </div>
    </div>

    <!-- Navigation Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
      <!-- Traits Card -->
      <router-link
        to="/traits"
        class="bg-blue-50 hover:bg-blue-100 p-6 rounded-lg transition-colors"
      >
        <div class="text-blue-600 text-3xl mb-3">🧬</div>
        <h3 class="font-semibold text-lg mb-2">Explore Traits</h3>
        <p class="text-gray-600 text-sm">
          Browse and search trait labels from MR studies
        </p>
      </router-link>

      <!-- Studies Card -->
      <router-link
        to="/studies"
        class="bg-green-50 hover:bg-green-100 p-6 rounded-lg transition-colors"
      >
        <div class="text-green-600 text-3xl mb-3">📋</div>
        <h3 class="font-semibold text-lg mb-2">Study Details</h3>
        <p class="text-gray-600 text-sm">
          View study metadata and analysis results
        </p>
      </router-link>

      <!-- Similarities Card -->
      <router-link
        to="/similarities"
        class="bg-purple-50 hover:bg-purple-100 p-6 rounded-lg transition-colors"
      >
        <div class="text-purple-600 text-3xl mb-3">🔗</div>
        <h3 class="font-semibold text-lg mb-2">Similarities</h3>
        <p class="text-gray-600 text-sm">
          Analyze trait and study similarity relationships
        </p>
      </router-link>

      <!-- About Card -->
      <router-link
        to="/about"
        class="bg-orange-50 hover:bg-orange-100 p-6 rounded-lg transition-colors"
      >
        <div class="text-orange-600 text-3xl mb-3">ℹ️</div>
        <h3 class="font-semibold text-lg mb-2">About</h3>
        <p class="text-gray-600 text-sm">
          Learn about the project and methodology
        </p>
      </router-link>
    </div>

    <!-- Top Traits Section -->
    <div v-if="overview && overview.top_traits.length > 0" class="mb-12">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold text-gray-900">Top Traits</h2>
        <router-link
          to="/traits"
          class="text-blue-600 hover:text-blue-500 font-medium"
        >
          View all traits →
        </router-link>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div
          v-for="trait in overview.top_traits.slice(0, 6)"
          :key="trait.trait_index"
          class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow duration-200"
        >
          <div class="flex items-center justify-between">
            <div class="flex-1 min-w-0">
              <h3
                class="text-sm font-medium text-gray-900 truncate"
                :title="trait.trait_label"
              >
                {{ trait.trait_label }}
              </h3>
              <p class="text-xs text-gray-500 mt-1">
                Index: {{ trait.trait_index }}
              </p>
            </div>
            <div class="ml-4 flex-shrink-0 text-right">
              <div class="text-lg font-bold text-blue-600">
                {{ trait.appearance_count }}
              </div>
              <div class="text-xs text-gray-500">appearances</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Model Coverage Section -->
    <div v-if="overview && overview.model_coverage.length > 0" class="mb-12">
      <h2 class="text-2xl font-bold text-gray-900 mb-6">Model Coverage</h2>

      <div class="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">
            Trait Distribution by Model
          </h3>
        </div>
        <div class="divide-y divide-gray-200">
          <div
            v-for="model in overview.model_coverage"
            :key="model.model"
            class="px-6 py-4 flex items-center justify-between"
          >
            <div>
              <div class="font-medium text-gray-900">{{ model.model }}</div>
              <div class="text-sm text-gray-500">
                {{ model.total_mentions.toLocaleString() }} total mentions
              </div>
            </div>
            <div class="text-right">
              <div class="text-lg font-bold text-gray-900">
                {{ model.unique_traits.toLocaleString() }}
              </div>
              <div class="text-sm text-gray-500">unique traits</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Key Features Section -->
    <div class="mb-12">
      <h2 class="text-2xl font-bold text-gray-900 mb-6">Key Features</h2>
      <div class="grid md:grid-cols-2 gap-6">
        <div class="bg-white border border-gray-200 rounded-lg p-6">
          <h3 class="font-semibold text-lg mb-3">Vector Search</h3>
          <p class="text-gray-600">
            Find similar traits and studies using semantic vector embeddings for
            advanced similarity matching.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-6">
          <h3 class="font-semibold text-lg mb-3">LLM Extraction</h3>
          <p class="text-gray-600">
            Data extracted from PubMed literature using multiple large language
            models for comprehensive coverage.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-6">
          <h3 class="font-semibold text-lg mb-3">Interactive Exploration</h3>
          <p class="text-gray-600">
            Modern web interface for filtering, searching, and analyzing
            Mendelian Randomization study data.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-6">
          <h3 class="font-semibold text-lg mb-3">Research Integration</h3>
          <p class="text-gray-600">
            Built for researchers to discover relationships between traits and
            explore MR study methodology.
          </p>
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="isLoading" class="text-center py-12">
      <LoadingSpinner size="lg" text="Loading overview data..." />
    </div>

    <!-- Error State -->
    <ErrorAlert
      v-if="error"
      :error="error"
      variant="error"
      class="mb-6"
      @dismiss="error = undefined"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

import { useTraitsStore } from '@/stores/traits'
import type { TraitsOverview } from '@/types/api'

// Components
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import ErrorAlert from '@/components/common/ErrorAlert.vue'

// ==== Setup ====

const traitsStore = useTraitsStore()

// ==== State ====

const overview = ref<TraitsOverview | null>(null)
const isLoading = ref(false)
const error = ref<string | undefined>(undefined)

// ==== Methods ====

async function loadOverview() {
  isLoading.value = true
  error.value = undefined

  try {
    await traitsStore.fetchTraitsOverview()
    overview.value = traitsStore.traitsOverview
  } catch (err) {
    console.error('Failed to load overview:', err)
    error.value =
      err instanceof Error ? err.message : 'Failed to load overview data'
  } finally {
    isLoading.value = false
  }
}

// ==== Lifecycle ====

onMounted(() => {
  loadOverview()
})
</script>
