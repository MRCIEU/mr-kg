<template>
  <div
    class="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow duration-200"
    :class="cardClass"
    data-testid="trait-card"
    @click="handleCardClick"
  >
    <!-- Card header -->
    <div class="p-4 border-b border-gray-100">
      <div class="flex items-start justify-between">
        <div class="flex-1 min-w-0">
          <h3
            class="text-lg font-medium text-gray-900 truncate"
            :title="trait.trait_label"
            data-testid="trait-label"
          >
            {{ trait.trait_label }}
          </h3>
          <p class="text-sm text-gray-500 mt-1">
            Index: {{ trait.trait_index }}
          </p>
        </div>

        <!-- Selection indicator -->
        <div v-if="selectable" class="ml-3 flex-shrink-0">
          <input
            :id="`trait-${trait.trait_index}`"
            type="checkbox"
            :checked="isSelected"
            class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            @change="handleSelectionChange"
          />
        </div>

        <!-- Action menu -->
        <div v-else-if="showActions" class="ml-3 flex-shrink-0">
          <button
            type="button"
            class="text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-full p-1"
            @click="$emit('menu-click', trait)"
          >
            <EllipsisVerticalIcon class="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>

    <!-- Card content -->
    <div class="p-4">
      <!-- Appearance count -->
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center">
          <ChartBarIcon class="h-5 w-5 text-gray-400 mr-2" aria-hidden="true" />
          <span class="text-sm font-medium text-gray-900">
            {{ trait.appearance_count.toLocaleString() }}
          </span>
          <span class="text-sm text-gray-500 ml-1">appearances</span>
        </div>

        <!-- Appearance count badge -->
        <span :class="appearanceBadgeClass">
          {{ getAppearanceLevel() }}
        </span>
      </div>

      <!-- Statistics (if available) -->
      <div
        v-if="showStats && statistics"
        class="space-y-3 mb-4"
        data-testid="trait-statistics"
      >
        <div class="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span class="text-gray-500">Studies:</span>
            <span class="font-medium ml-1">{{
              statistics.study_count || 0
            }}</span>
          </div>
          <div>
            <span class="text-gray-500">Models:</span>
            <span class="font-medium ml-1">{{
              Object.keys(statistics.model_distribution || {}).length
            }}</span>
          </div>
        </div>

        <!-- Model distribution -->
        <div v-if="statistics.model_distribution" class="space-y-1">
          <h4 class="text-xs font-medium text-gray-700 uppercase tracking-wide">
            Model Distribution
          </h4>
          <div class="space-y-1">
            <div
              v-for="[model, count] in Object.entries(
                statistics.model_distribution
              ).slice(0, 3)"
              :key="model"
              class="flex justify-between text-xs"
            >
              <span class="text-gray-600">{{ model }}</span>
              <span class="font-medium">{{ count }}</span>
            </div>
            <div
              v-if="Object.keys(statistics.model_distribution).length > 3"
              class="text-xs text-gray-400"
            >
              +{{ Object.keys(statistics.model_distribution).length - 3 }} more
            </div>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div
        class="flex items-center justify-between pt-3 border-t border-gray-100"
      >
        <div class="flex space-x-2">
          <button
            type="button"
            class="inline-flex items-center px-3 py-1.5 border border-gray-300 text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            @click="$emit('view-details', trait)"
          >
            <EyeIcon class="h-3 w-3 mr-1" aria-hidden="true" />
            Details
          </button>

          <button
            v-if="showSimilarAction"
            type="button"
            class="inline-flex items-center px-3 py-1.5 border border-gray-300 text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            @click="$emit('find-similar', trait)"
          >
            <MagnifyingGlassIcon class="h-3 w-3 mr-1" aria-hidden="true" />
            Similar
          </button>
        </div>

        <!-- Compact mode toggle -->
        <button
          v-if="allowCompactToggle"
          type="button"
          class="text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-full p-1"
          @click="toggleCompact"
          :title="isCompact ? 'Expand' : 'Collapse'"
        >
          <ChevronUpIcon v-if="!isCompact" class="h-4 w-4" aria-hidden="true" />
          <ChevronDownIcon v-else class="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </div>

    <!-- Compact mode content -->
    <div v-if="isCompact" class="px-4 pb-4 border-t border-gray-100">
      <div class="flex items-center justify-between text-sm">
        <span class="text-gray-500">{{ trait.trait_label }}</span>
        <span class="font-medium">{{
          trait.appearance_count.toLocaleString()
        }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import {
  ChartBarIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  EllipsisVerticalIcon,
  EyeIcon,
  MagnifyingGlassIcon,
} from '@heroicons/vue/24/outline'
import type { TraitListItem, TraitStats } from '@/types/api'

// ==== Props ====

interface Props {
  /** Trait data */
  trait: TraitListItem
  /** Additional statistics */
  statistics?: TraitStats
  /** Whether the card can be selected */
  selectable?: boolean
  /** Whether the trait is selected */
  isSelected?: boolean
  /** Show actions menu */
  showActions?: boolean
  /** Show statistics section */
  showStats?: boolean
  /** Show similar traits action */
  showSimilarAction?: boolean
  /** Allow compact mode toggle */
  allowCompactToggle?: boolean
  /** Card size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Whether card is clickable */
  clickable?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  selectable: false,
  isSelected: false,
  showActions: false,
  showStats: true,
  showSimilarAction: true,
  allowCompactToggle: false,
  size: 'md',
  clickable: true,
})

// ==== Emits ====

const emit = defineEmits<{
  'view-details': [trait: TraitListItem]
  'find-similar': [trait: TraitListItem]
  'selection-change': [trait: TraitListItem, selected: boolean]
  'menu-click': [trait: TraitListItem]
  click: [trait: TraitListItem]
}>()

// ==== State ====

const isCompact = ref(false)

// ==== Computed ====

const cardClass = computed(() => {
  const classes = []

  if (props.clickable) {
    classes.push('cursor-pointer')
  }

  if (props.isSelected) {
    classes.push('ring-2 ring-blue-500 border-blue-300')
  }

  return classes.join(' ')
})

const appearanceBadgeClass = computed(() => {
  const level = getAppearanceLevel()
  const baseClasses = 'px-2 py-1 text-xs font-medium rounded-full'

  switch (level) {
    case 'Very High':
      return `${baseClasses} bg-red-100 text-red-800`
    case 'High':
      return `${baseClasses} bg-orange-100 text-orange-800`
    case 'Medium':
      return `${baseClasses} bg-yellow-100 text-yellow-800`
    case 'Low':
      return `${baseClasses} bg-green-100 text-green-800`
    default:
      return `${baseClasses} bg-gray-100 text-gray-800`
  }
})

// ==== Methods ====

function getAppearanceLevel(): string {
  const count = props.trait.appearance_count

  if (count >= 100) return 'Very High'
  if (count >= 50) return 'High'
  if (count >= 10) return 'Medium'
  if (count >= 2) return 'Low'
  return 'Rare'
}

function handleSelectionChange(event: Event) {
  const target = event.target as HTMLInputElement
  emit('selection-change', props.trait, target.checked)
}

function toggleCompact() {
  isCompact.value = !isCompact.value
}

function handleCardClick() {
  if (props.clickable) {
    emit('click', props.trait)
  }
}
</script>
