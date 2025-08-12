<template>
  <div class="px-4 py-6 sm:px-0">
    <div class="mb-6">
      <h1 class="text-3xl font-bold text-gray-900">Explore Traits</h1>
      <p class="mt-2 text-gray-600">
        Browse and filter trait labels extracted from MR literature data.
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Left Column: Trait List -->
      <div class="bg-white rounded-lg shadow">
        <div class="p-6">
          <h2 class="text-lg font-medium text-gray-900 mb-4">Trait Labels</h2>
          
          <!-- Filter Input -->
          <div class="mb-4">
            <input
              v-model="filterText"
              type="text"
              placeholder="Filter trait labels..."
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              @input="debouncedSearch"
            />
          </div>

          <!-- Results Summary -->
          <div class="mb-4 text-sm text-gray-600">
            <span v-if="filterText">
              Showing {{ traits.length }} of {{ totalCount }} traits
            </span>
            <span v-else>
              Showing top {{ traits.length }} of {{ totalCount }} traits
            </span>
          </div>

          <!-- Trait List -->
          <div class="space-y-2 max-h-96 overflow-y-auto">
            <div
              v-for="trait in traits"
              :key="trait.trait_label"
              class="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 cursor-pointer"
              :class="{ 'bg-indigo-50 border-indigo-200': selectedTrait === trait.trait_label }"
              @click="selectTrait(trait.trait_label)"
            >
              <div class="flex-1">
                <div class="font-medium text-gray-900">
                  {{ trait.trait_label }}
                </div>
                <div class="text-sm text-gray-500">
                  {{ trait.appearance_count }} occurrences
                </div>
              </div>
              <button
                class="ml-3 px-3 py-1 text-xs font-medium rounded-md"
                :class="selectedTrait === trait.trait_label 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'"
              >
                {{ selectedTrait === trait.trait_label ? '✓ Selected' : 'Select' }}
              </button>
            </div>
          </div>

          <!-- No Results Message -->
          <div v-if="traits.length === 0 && !isLoading" class="text-center py-8 text-gray-500">
            <div class="text-4xl mb-2">🔍</div>
            <p>No traits found matching "{{ filterText }}"</p>
            <p class="text-sm mt-1">Try adjusting your search terms</p>
          </div>

          <!-- Loading State -->
          <div v-if="isLoading" class="text-center py-8 text-gray-500">
            <div class="text-4xl mb-2">⏳</div>
            <p>Loading traits...</p>
          </div>
        </div>
      </div>

      <!-- Right Column: Selected Trait Details -->
      <div class="bg-white rounded-lg shadow">
        <div class="p-6">
          <h2 class="text-lg font-medium text-gray-900 mb-4">Trait Details</h2>
          
          <div v-if="selectedTrait" class="space-y-4">
            <div>
              <h3 class="text-xl font-semibold text-gray-900">{{ selectedTrait }}</h3>
              <button
                @click="clearSelection"
                class="mt-2 text-sm text-indigo-600 hover:text-indigo-500"
              >
                Clear Selection
              </button>
            </div>

            <!-- Model Selection -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                Select Model:
              </label>
              <select
                v-model="selectedModel"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                @change="fetchStudies"
              >
                <option value="">Choose a model...</option>
                <option v-for="model in models" :key="model" :value="model">
                  {{ model }}
                </option>
              </select>
            </div>

            <!-- Studies List -->
            <div v-if="selectedModel && studies.length > 0" class="space-y-3">
              <h4 class="font-medium text-gray-900">
                Studies ({{ studies.length }} results)
              </h4>
              <div class="space-y-2 max-h-64 overflow-y-auto">
                <div
                  v-for="study in studies"
                  :key="study.model_result_id"
                  class="p-3 border rounded-lg bg-gray-50"
                >
                  <div class="font-medium text-sm text-gray-900">
                    PMID: {{ study.pmid }}
                  </div>
                  <div v-if="study.title" class="text-sm text-gray-600 mt-1">
                    {{ study.title }}
                  </div>
                  <div v-if="study.journal" class="text-xs text-gray-500 mt-1">
                    {{ study.journal }}
                    <span v-if="study.pub_date"> • {{ study.pub_date }}</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- No Studies Message -->
            <div v-else-if="selectedModel && studies.length === 0 && !isLoadingStudies" class="text-center py-4 text-gray-500">
              <p>No studies found for this trait in {{ selectedModel }}</p>
            </div>

            <!-- Loading Studies -->
            <div v-if="isLoadingStudies" class="text-center py-4 text-gray-500">
              <p>Loading studies...</p>
            </div>
          </div>

          <div v-else class="text-center py-8 text-gray-500">
            <div class="text-4xl mb-2">👆</div>
            <p>Select a trait from the list to view details</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useTraitsStore } from '../stores/traits'

const traitsStore = useTraitsStore()

// Reactive data
const filterText = ref('')
const selectedTrait = ref<string | null>(null)
const selectedModel = ref('')
const isLoading = ref(false)
const isLoadingStudies = ref(false)

// Computed properties from store
const traits = ref(traitsStore.traits)
const totalCount = ref(traitsStore.totalCount)
const models = ref(traitsStore.models)
const studies = ref(traitsStore.studies)

// Debounced search
let searchTimeout: NodeJS.Timeout
const debouncedSearch = () => {
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(() => {
    searchTraits()
  }, 300)
}

// Methods
const searchTraits = async () => {
  isLoading.value = true
  await traitsStore.searchTraits(filterText.value)
  traits.value = traitsStore.traits
  totalCount.value = traitsStore.totalCount
  isLoading.value = false
}

const selectTrait = (traitLabel: string) => {
  if (selectedTrait.value === traitLabel) {
    clearSelection()
  } else {
    selectedTrait.value = traitLabel
    selectedModel.value = ''
    studies.value = []
  }
}

const clearSelection = () => {
  selectedTrait.value = null
  selectedModel.value = ''
  studies.value = []
}

const fetchStudies = async () => {
  if (!selectedTrait.value || !selectedModel.value) return
  
  isLoadingStudies.value = true
  await traitsStore.fetchStudiesForTrait(selectedTrait.value, selectedModel.value)
  studies.value = traitsStore.studies
  isLoadingStudies.value = false
}

// Lifecycle
onMounted(async () => {
  isLoading.value = true
  await Promise.all([
    traitsStore.fetchTopTraits(),
    traitsStore.fetchModels()
  ])
  traits.value = traitsStore.traits
  totalCount.value = traitsStore.totalCount
  models.value = traitsStore.models
  isLoading.value = false
})
</script>
