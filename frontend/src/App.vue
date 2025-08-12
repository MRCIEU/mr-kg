<template>
  <div id="app" class="min-h-screen bg-gray-50">
    <nav class="bg-white shadow-sm border-b">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <router-link to="/" class="text-xl font-bold text-gray-900">
              MR-KG Explorer
            </router-link>
          </div>
          <div class="flex space-x-8">
            <router-link
              to="/"
              class="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 border-b-2 border-transparent hover:border-gray-300"
              :class="{ 'border-indigo-500': $route.path === '/' }"
            >
              Home
            </router-link>
            <router-link
              to="/explore-traits"
              class="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 border-b-2 border-transparent hover:border-gray-300"
              :class="{ 'border-indigo-500': $route.path === '/explore-traits' }"
            >
              Explore Traits
            </router-link>
          </div>
        </div>
      </div>
    </nav>

    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
// Component logic here
</script>
