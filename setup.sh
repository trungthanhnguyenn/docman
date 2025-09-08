#!/bin/bash

# Setup script for Document Management System
echo "🚀 Setting up Document Management System"
echo "=" * 50

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. Please review and update the values if needed."
fi

# Start services
echo "🐳 Starting database services with Docker Compose..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
echo "   This may take a few minutes on first run..."

# Wait for PostgreSQL
echo "🔄 Waiting for PostgreSQL..."
until docker compose exec -T postgres pg_isready -U postgres; do
    sleep 2
done
echo "✅ PostgreSQL is ready!"

# Wait for MinIO
echo "🔄 Waiting for MinIO..."
until curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; do
    sleep 2
done
echo "✅ MinIO is ready!"

# Wait for Qdrant
echo "🔄 Waiting for Qdrant..."
until curl -sf http://localhost:6333/health > /dev/null 2>&1; do
    sleep 2
done
echo "✅ Qdrant is ready!"

echo ""
echo "🎉 All services are ready!"
echo ""
echo "📋 Service URLs:"
echo "   - PostgreSQL: localhost:5432"
echo "   - MinIO: http://localhost:9000 (UI: http://localhost:9001)"
echo "   - Qdrant: http://localhost:6333"
echo ""
echo "🔑 Default credentials:"
echo "   - PostgreSQL: postgres/postgres"
echo "   - MinIO: minioadmin/minioadmin"
echo ""
echo "🚀 You can now start the API server:"
echo "   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "🧪 Run tests with:"
echo "   python test_session_api.py"
