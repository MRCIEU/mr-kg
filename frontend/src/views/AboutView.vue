<template>
  <div class="max-w-4xl mx-auto">
    <h1 class="text-3xl font-bold text-gray-900 mb-8">About MR-KG</h1>

    <div class="prose prose-lg max-w-none">
      <p class="text-xl text-gray-600 mb-8">
        MR-KG (Mendelian Randomization Knowledge Graph) is a comprehensive
        system for exploring structural PubMed literature data extracted by
        large language models.
      </p>

      <h2>Project Overview</h2>
      <p>
        This project processes and analyzes Mendelian Randomization studies from
        PubMed literature using advanced language models. The system builds
        vector databases from LLM-extracted trait information and provides an
        interactive web interface for exploration and analysis.
      </p>

      <h2>Key Features</h2>
      <ul>
        <li>Vector similarity search for trait and study relationships</li>
        <li>Multi-model LLM extraction for comprehensive coverage</li>
        <li>Interactive filtering and exploration tools</li>
        <li>Real-time similarity computation</li>
        <li>Comprehensive trait and study metadata</li>
      </ul>

      <h2>Technology Stack</h2>
      <ul>
        <li><strong>Backend</strong>: FastAPI with DuckDB vector storage</li>
        <li><strong>Frontend</strong>: Vue.js 3 with TypeScript</li>
        <li><strong>Data Processing</strong>: Python with spaCy embeddings</li>
        <li>
          <strong>Infrastructure</strong>: Docker with development environment
        </li>
      </ul>

      <h2>Data Sources</h2>
      <p>
        The knowledge graph is built from PubMed literature processed by
        multiple large language models including GPT-4, Claude, and Llama
        models. The system extracts structured information about Mendelian
        Randomization studies, including exposure and outcome traits,
        statistical measures, and study metadata.
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
// About view component logic
</script>
