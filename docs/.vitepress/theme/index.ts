import Theme from 'vitepress/theme'
import './styles.css'
import HomeCategories from './components/HomeCategories.vue'
import CustomHero from './components/CustomHero.vue'

export default {
  extends: Theme,
  enhanceApp({ app }) {
    app.component('HomeCategories', HomeCategories)
    app.component('CustomHero', CustomHero)
  }
}
