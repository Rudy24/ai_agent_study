import { createRouter, createWebHistory } from "vue-router";
import ChatView from "../views/ChatView.vue";
import DocsView from "../views/DocsView.vue";

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "chat", component: ChatView },
    { path: "/docs", name: "docs", component: DocsView },
  ],
});
