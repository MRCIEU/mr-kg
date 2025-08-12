<template>
  <div class="px-4 py-6 sm:px-0">
    <div class="text-center">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">
        MR-KG Literature Explorer
      </h1>
      <p class="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
        Explore and analyze Mendelian Randomization literature data through an
        interactive knowledge graph interface.
      </p>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
        <div class="bg-white p-6 rounded-lg shadow">
          <h2 class="text-xl font-semibold text-gray-900 mb-2">
            Trait Exploration
          </h2>
          <p class="text-gray-600 mb-4">
            Browse and filter trait labels extracted from MR literature.
          </p>
          <router-link
            to="/explore-traits"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
          >
            Explore Traits
          </router-link>
        </div>
        
        <div class="bg-white p-6 rounded-lg shadow">
          <h2 class="text-xl font-semibold text-gray-900 mb-2">
            Study Analysis
          </h2>
          <p class="text-gray-600 mb-4">
            Analyze studies and their associated traits across different models.
          </p>
          <button
            disabled
            class="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-500 bg-gray-100 cursor-not-allowed"
          >
            Coming Soon
          </button>
        </div>
      </div>

      <div class="mt-12">
        <h2 class="text-2xl font-bold text-gray-900 mb-4">Features</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <div class="text-center">
            <div class="text-indigo-600 text-3xl mb-2">🔍</div>
            <h3 class="text-lg font-medium text-gray-900">Interactive Search</h3>
            <p class="text-gray-600">Filter and search through trait labels with real-time results.</p>
          </div>
          <div class="text-center">
            <div class="text-indigo-600 text-3xl mb-2">📊</div>
            <h3 class="text-lg font-medium text-gray-900">Model Comparison</h3>
            <p class="text-gray-600">Compare results across different language models and approaches.</p>
          </div>
          <div class="text-center">
            <div class="text-indigo-600 text-3xl mb-2">📚</div>
            <h3 class="text-lg font-medium text-gray-900">Literature Analysis</h3>
            <p class="text-gray-600">Explore PubMed abstracts and their extracted trait information.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Component logic
</script>
