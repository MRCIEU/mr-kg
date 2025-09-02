import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/traits',
      name: 'traits',
      component: () => import('../views/TraitsView.vue'),
    },
    {
      path: '/studies',
      name: 'studies',
      component: () => import('../views/StudiesView.vue'),
    },
    {
      path: '/similarities',
      name: 'similarities',
      component: () => import('../views/SimilaritiesView.vue'),
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue'),
    },
  ],
})

export default router
