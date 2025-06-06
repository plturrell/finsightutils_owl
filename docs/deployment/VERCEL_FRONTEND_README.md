# Vercel Frontend for OWL Converter

This document provides instructions for setting up and deploying the frontend application for the OWL Converter system using Vercel.

## Prerequisites

- [Node.js](https://nodejs.org/) (v16 or later)
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/)
- [Vercel Account](https://vercel.com/signup)
- [Vercel CLI](https://vercel.com/cli) (optional, for command-line deployment)

## Architecture

The frontend is a Next.js application that:

1. Provides a user interface for converting SAP HANA schemas to OWL ontologies
2. Visualizes schema knowledge graphs using D3.js
3. Enables querying schema knowledge using natural language
4. Connects to the NVIDIA GPU-accelerated backend API

## Project Setup

### 1. Install Dependencies

```bash
# Navigate to the frontend directory
cd owl-frontend

# Install dependencies
npm install
# or
yarn install
```

### 2. Configure Environment Variables

Create or update `.env.local` for development:

```
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

For production deployment, create `.env.production`:

```
BACKEND_URL=https://your-backend-url.com
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.com
```

### 3. Run Development Server

```bash
npm run dev
# or
yarn dev
```

Visit `http://localhost:3000` to access the development server.

## Deployment to Vercel

### Option 1: Deploy via Vercel CLI

```bash
# Install Vercel CLI if not already installed
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

### Option 2: Deploy via Vercel Dashboard

1. Push your code to a Git repository (GitHub, GitLab, or Bitbucket)
2. Log in to the [Vercel Dashboard](https://vercel.com/dashboard)
3. Click "New Project"
4. Import your repository
5. Configure environment variables:
   - `BACKEND_URL`: URL of your NVIDIA backend API
   - `NEXT_PUBLIC_BACKEND_URL`: Same URL (for client-side access)
6. Click "Deploy"

### Option 3: Deploy with Custom Script

Use the provided deployment script:

```bash
./deploy_frontend.sh https://your-backend-url.com
```

## Configuration Files

### Next.js Configuration (`next.config.js`)

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.BACKEND_URL + '/:path*',
      },
    ];
  },
  images: {
    domains: ['localhost', 'your-backend-domain.com'],
  },
};

module.exports = nextConfig;
```

### Vercel Configuration (`vercel.json`)

```json
{
  "name": "owl-frontend",
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ],
  "routes": [
    {
      "src": "/api/v1/(.*)",
      "dest": "https://your-backend-url.com/api/v1/$1"
    }
  ],
  "env": {
    "BACKEND_URL": "https://your-backend-url.com"
  }
}
```

## Project Structure

```
owl-frontend/
├── components/         # React components
├── pages/              # Next.js pages
│   ├── api/            # API routes for backend proxying
│   ├── _app.js         # Application wrapper
│   ├── index.js        # Home page
│   └── visualize/      # Schema visualization page
├── public/             # Static assets
├── styles/             # CSS styles
├── .env.local          # Development environment variables
├── .env.production     # Production environment variables
├── next.config.js      # Next.js configuration
├── package.json        # Project dependencies
├── tailwind.config.js  # Tailwind CSS configuration
└── vercel.json         # Vercel deployment configuration
```

## Key Features

### 1. Schema Conversion

The home page provides a form to convert SAP HANA schemas to OWL ontologies.

Endpoint: `POST /api/v1/sap/owl/convert`

```javascript
const handleConvert = async () => {
  const response = await axios.post('/api/v1/sap/owl/convert', {
    schema_name: schema,
    inference_level: 'standard',
    force_refresh: false,
    batch_size: 50,
    chunked_processing: true
  });
};
```

### 2. Schema Visualization

The `/visualize` page renders a dynamic force-directed graph of the schema relationships.

Endpoint: `GET /api/v1/sap/owl/knowledge-graph/{schema_name}`

Technologies:
- D3.js for visualization
- SWR for data fetching and caching
- React hooks for state management

### 3. Natural Language Querying

Query schema knowledge using natural language.

Endpoint: `POST /api/v1/sap/owl/query`

```javascript
const handleQuery = async () => {
  const response = await axios.post('/api/v1/sap/owl/query', {
    schema_name: schema,
    query: 'List all tables with foreign key relationships'
  });
};
```

### 4. SQL Translation

Translate natural language to SQL.

Endpoint: `POST /api/v1/sap/owl/translate`

```javascript
const handleTranslate = async () => {
  const response = await axios.post('/api/v1/sap/owl/translate', {
    schema_name: schema,
    query: 'Show me all customers with more than 5 orders'
  });
};
```

## API Proxying

The frontend proxies API requests to the backend through:

1. **Development**: Next.js API routes in `pages/api/`
2. **Production**: Vercel rewrites configured in `vercel.json`

This ensures secure cross-origin requests and simplifies frontend code.

## Styling and UI Components

The project uses:

- Tailwind CSS for styling
- React components for UI elements
- Responsive design for mobile and desktop

## Performance Optimization

- **Image Optimization**: Next.js automatic image optimization
- **Code Splitting**: Dynamic imports for page components
- **Static Generation**: Pre-rendered pages where possible
- **Client-side Data Fetching**: SWR for efficient data loading

## Monitoring and Analytics

Add monitoring by:

1. **Vercel Analytics**: Enable in Vercel project settings
2. **Custom Logging**: Implement in API routes
3. **Error Tracking**: Integrate Sentry or similar

## Troubleshooting

### CORS Issues

If you encounter CORS problems:

1. Ensure backend has proper CORS headers
2. Verify API proxying configuration
3. Check network requests in browser developer tools

### API Connection Issues

If the frontend can't connect to the backend:

1. Verify environment variables are set correctly
2. Ensure backend is running and accessible
3. Check for network constraints or firewalls
4. Verify Vercel routing configuration

### Deployment Problems

Common deployment issues:

1. Environment variables not set in Vercel
2. Backend URL not accessible from Vercel
3. Next.js build errors

## Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [D3.js Documentation](https://d3js.org/)