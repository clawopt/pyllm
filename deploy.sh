#!/bin/bash
if [ "$1" != "--deploy-only" ]; then
  BASE_URL=/ npm run build --prefix docs
fi
npx wrangler pages deploy docs/.vitepress/dist
