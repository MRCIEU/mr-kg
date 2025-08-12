import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import ExploreTraits from '../views/ExploreTraits.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
    },
    {
      path: '/explore-traits',
      name: 'explore-traits',
      component: ExploreTraits,
    },
  ],
})

export default router
