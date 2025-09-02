<template>
  <div id="app">
    <nav class="bg-gray-800 text-white p-4">
      <div class="container mx-auto flex justify-between items-center">
        <router-link to="/" class="text-xl font-bold">
          MR-KG Explorer
        </router-link>
        <div class="space-x-4">
          <router-link to="/" class="hover:text-gray-300">Home</router-link>
          <router-link to="/traits" class="hover:text-gray-300"
            >Traits</router-link
          >
          <router-link to="/studies" class="hover:text-gray-300"
            >Studies</router-link
          >
          <router-link to="/similarities" class="hover:text-gray-300"
            >Similarities</router-link
          >
          <router-link to="/about" class="hover:text-gray-300"
            >About</router-link
          >
        </div>
      </div>
    </nav>

    <main class="container mx-auto px-4 py-8">
      <router-view />
    </main>

    <footer class="bg-gray-100 text-center py-4 mt-16">
      <p class="text-gray-600">
        &copy; 2025 MR-KG (Mendelian Randomization Knowledge Graph)
      </p>
    </footer>
  </div>
</template>

<script setup lang="ts">
// App component logic can be added here
</script>

<style scoped>
.router-link-active {
  @apply text-blue-300;
}
</style>
