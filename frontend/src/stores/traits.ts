import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../services/api'

export interface TraitLabel {
  trait_label: string
  appearance_count: number
}

export interface StudyResult {
  model_result_id: number
  pmid: string
  title?: string
  journal?: string
  pub_date?: string
  metadata?: any
}

export const useTraitsStore = defineStore('traits', () => {
  // State
  const traits = ref<TraitLabel[]>([])
  const totalCount = ref(0)
  const models = ref<string[]>([])
  const studies = ref<StudyResult[]>([])
  const isLoading = ref(false)

  // Actions
  const fetchTopTraits = async (limit: number = 50) => {
    try {
      isLoading.value = true
      const response = await api.get('/api/traits/top', {
        params: { limit }
      })
      traits.value = response.data.traits
      totalCount.value = response.data.total_count
    } catch (error) {
      console.error('Error fetching top traits:', error)
    } finally {
      isLoading.value = false
    }
  }

  const searchTraits = async (filterText: string, limit: number = 100) => {
    try {
      isLoading.value = true
      const response = await api.get('/api/traits/search', {
        params: { filter_text: filterText, limit }
      })
      traits.value = response.data.traits
      totalCount.value = response.data.total_count
    } catch (error) {
      console.error('Error searching traits:', error)
    } finally {
      isLoading.value = false
    }
  }

  const fetchModels = async () => {
    try {
      const response = await api.get('/api/models')
      models.value = response.data.models
    } catch (error) {
      console.error('Error fetching models:', error)
    }
  }

  const fetchStudiesForTrait = async (traitLabel: string, model: string, limit: number = 100) => {
    try {
      const response = await api.get(`/api/traits/${encodeURIComponent(traitLabel)}/studies`, {
        params: { model, limit }
      })
      studies.value = response.data.studies
    } catch (error) {
      console.error('Error fetching studies:', error)
    }
  }

  return {
    // State
    traits,
    totalCount,
    models,
    studies,
    isLoading,
    // Actions
    fetchTopTraits,
    searchTraits,
    fetchModels,
    fetchStudiesForTrait,
  }
})
