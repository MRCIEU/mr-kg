# Production Dockerfile for Vue.js Frontend
# Multi-stage build for optimized production image

# ==== Build Stage ====
FROM node:18-alpine as builder

# Install build dependencies
RUN apk add --no-cache git

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production --silent

# Copy source code
COPY . .

# Build application for production
RUN npm run build

# ==== Production Stage ====
FROM nginx:alpine as production

# Install runtime dependencies
RUN apk add --no-cache curl

# Create non-root user for security
RUN addgroup -g 1001 -S app && adduser -S app -u 1001 -G app

# Create nginx configuration
RUN cat > /etc/nginx/conf.d/default.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Gzip compression
    gzip on;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    # Handle client-side routing
    location / {
        try_files $uri $uri/ /index.html;

        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Create environment substitution script
RUN cat > /docker-entrypoint.d/30-substitute-env.sh << 'EOF'
#!/bin/sh
set -e

# Default values if environment variables are not set
API_BASE_URL=${VITE_API_BASE_URL:-"http://localhost:8000/api/v1"}
APP_TITLE=${VITE_APP_TITLE:-"MR-KG Explorer"}
APP_DESCRIPTION=${VITE_APP_DESCRIPTION:-"Mendelian Randomization Knowledge Graph"}

echo "Substituting environment variables in built files..."
echo "API_BASE_URL: $API_BASE_URL"
echo "APP_TITLE: $APP_TITLE"
echo "APP_DESCRIPTION: $APP_DESCRIPTION"

# Find and replace environment variables in JavaScript files
find /usr/share/nginx/html -name "*.js" -type f -exec sed -i \
    -e "s|VITE_API_BASE_URL_PLACEHOLDER|$API_BASE_URL|g" \
    -e "s|VITE_APP_TITLE_PLACEHOLDER|$APP_TITLE|g" \
    -e "s|VITE_APP_DESCRIPTION_PLACEHOLDER|$APP_DESCRIPTION|g" \
    {} \;

echo "Environment variable substitution completed."
EOF

# Make substitution script executable
RUN chmod +x /docker-entrypoint.d/30-substitute-env.sh

# Set proper permissions
RUN chown -R app:app /usr/share/nginx/html \
    && chmod -R 755 /usr/share/nginx/html

# Set environment variables
ENV NODE_ENV=production
ENV NGINX_ENVSUBST_OUTPUT_DIR=/etc/nginx/conf.d

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Use default nginx entrypoint which will run our substitution script
CMD ["nginx", "-g", "daemon off;"]
