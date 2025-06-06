#!/bin/bash
# Stage 3: Vercel Deployment and NVIDIA Backend Connection

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== STAGE 3: Vercel Deployment and NVIDIA Backend Connection ===${NC}"

# Check for required tools
echo -e "${YELLOW}Step 1: Checking required tools...${NC}"

# Check for Node.js
if command -v node &> /dev/null; then
    echo -e "${GREEN}Node.js is installed:${NC}"
    node --version
else
    echo -e "${RED}Error: Node.js is not installed. Please install Node.js and try again.${NC}"
    echo "Visit https://nodejs.org/ for installation instructions."
    exit 1
fi

# Check for npm
if command -v npm &> /dev/null; then
    echo -e "${GREEN}npm is installed:${NC}"
    npm --version
else
    echo -e "${RED}Error: npm is not installed. Please install Node.js (includes npm) and try again.${NC}"
    exit 1
fi

# Check for Vercel CLI
if command -v vercel &> /dev/null; then
    echo -e "${GREEN}Vercel CLI is installed:${NC}"
    vercel --version
else
    echo -e "${YELLOW}Vercel CLI is not installed. Installing now...${NC}"
    npm install -g vercel
    
    # Check if installation was successful
    if ! command -v vercel &> /dev/null; then
        echo -e "${RED}Error: Failed to install Vercel CLI. Please install manually with 'npm install -g vercel'.${NC}"
        exit 1
    fi
fi

# Create frontend directory
echo -e "${YELLOW}Step 2: Creating frontend project...${NC}"
FRONTEND_DIR="owl-frontend"

# Check if directory already exists
if [ -d "$FRONTEND_DIR" ]; then
    echo -e "${YELLOW}Frontend directory already exists. Do you want to overwrite it?${NC}"
    read -p "Overwrite $FRONTEND_DIR? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$FRONTEND_DIR"
    else
        echo -e "${YELLOW}Using existing directory.${NC}"
    fi
fi

# Create directory if it doesn't exist
if [ ! -d "$FRONTEND_DIR" ]; then
    mkdir -p "$FRONTEND_DIR"
    
    # Initialize Next.js project
    echo -e "${YELLOW}Initializing Next.js project...${NC}"
    cd "$FRONTEND_DIR"
    
    # Create package.json
    cat > package.json << 'EOF'
{
  "name": "owl-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^13.4.19",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.4.0",
    "d3": "^7.8.5",
    "swr": "^2.2.2",
    "tailwindcss": "^3.3.3",
    "autoprefixer": "^10.4.15",
    "postcss": "^8.4.29"
  },
  "devDependencies": {
    "@types/node": "^20.5.7",
    "@types/react": "^18.2.21",
    "eslint": "^8.48.0",
    "eslint-config-next": "^13.4.19",
    "typescript": "^5.2.2"
  }
}
EOF
    
    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
    
    # Create Next.js configuration
    echo -e "${YELLOW}Creating Next.js configuration...${NC}"
    
    # Create next.config.js
    cat > next.config.js << 'EOF'
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
    domains: ['localhost'],
  },
};

module.exports = nextConfig;
EOF
    
    # Create .env.local
    cat > .env.local << 'EOF'
# Backend URL - update with your NVIDIA backend URL
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
EOF
    
    # Create .env.production
    cat > .env.production << 'EOF'
# Backend URL - update with your production NVIDIA backend URL
BACKEND_URL=https://your-backend-url.com
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.com
EOF
    
    # Create directories
    mkdir -p pages/api components public styles
    
    # Create tailwind.config.js
    cat > tailwind.config.js << 'EOF'
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
EOF
    
    # Create postcss.config.js
    cat > postcss.config.js << 'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF
    
    # Create _app.js
    mkdir -p pages
    cat > pages/_app.js << 'EOF'
import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />;
}

export default MyApp;
EOF
    
    # Create global CSS
    mkdir -p styles
    cat > styles/globals.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

html,
body {
  padding: 0;
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen,
    Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;
}

a {
  color: inherit;
  text-decoration: none;
}

* {
  box-sizing: border-box;
}
EOF
    
    # Create index.js
    cat > pages/index.js << 'EOF'
import { useState } from 'react';
import Head from 'next/head';
import axios from 'axios';

export default function Home() {
  const [schema, setSchema] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  
  const handleConvert = async () => {
    if (!schema) {
      setError('Please enter a schema name');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/v1/sap/owl/convert', {
        schema_name: schema,
        inference_level: 'standard',
        force_refresh: false,
        batch_size: 50,
        chunked_processing: true
      });
      
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleQuerySchema = async () => {
    if (!schema) {
      setError('Please enter a schema name');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/v1/sap/owl/query', {
        schema_name: schema,
        query: 'List all tables'
      });
      
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>SAP HANA to OWL Converter</title>
        <meta name="description" content="Convert SAP HANA schemas to OWL ontologies" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-center mb-8">
          SAP HANA to OWL Converter
        </h1>
        
        <div className="bg-white p-6 rounded-lg shadow-md max-w-2xl mx-auto">
          <div className="mb-4">
            <label htmlFor="schema" className="block text-gray-700 mb-2">
              Schema Name
            </label>
            <input
              type="text"
              id="schema"
              value={schema}
              onChange={(e) => setSchema(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter SAP HANA schema name"
            />
          </div>
          
          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
              {error}
            </div>
          )}
          
          <div className="flex space-x-4">
            <button
              onClick={handleConvert}
              disabled={loading}
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {loading ? 'Converting...' : 'Convert to OWL'}
            </button>
            
            <button
              onClick={handleQuerySchema}
              disabled={loading}
              className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
            >
              {loading ? 'Querying...' : 'Query Schema'}
            </button>
          </div>
          
          {result && (
            <div className="mt-6">
              <h2 className="text-xl font-semibold mb-3">Result</h2>
              <pre className="bg-gray-100 p-4 rounded-md overflow-auto max-h-96">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </main>
      
      <footer className="py-6 text-center text-gray-600">
        <p>Powered by NVIDIA GPU Acceleration</p>
      </footer>
    </div>
  );
}
EOF
    
    # Create API proxy for schema visualization
    mkdir -p pages/api
    cat > pages/api/knowledge-graph.js << 'EOF'
export default async function handler(req, res) {
  const { schema_name } = req.query;
  
  if (!schema_name) {
    return res.status(400).json({ error: 'Schema name is required' });
  }
  
  try {
    // Get the backend URL from environment variables
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    // Forward request to the backend
    const response = await fetch(`${backendUrl}/api/v1/sap/owl/knowledge-graph/${schema_name}`);
    
    if (!response.ok) {
      throw new Error(`Backend error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return res.status(200).json(data);
  } catch (error) {
    console.error('Error in knowledge-graph API:', error);
    return res.status(500).json({ error: error.message });
  }
}
EOF
    
    # Create schema visualization page
    mkdir -p pages/visualize
    cat > pages/visualize/index.js << 'EOF'
import { useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import axios from 'axios';
import * as d3 from 'd3';

export default function SchemaVisualizer() {
  const [schema, setSchema] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const svgRef = useRef(null);
  
  const handleVisualize = async () => {
    if (!schema) {
      setError('Please enter a schema name');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.get(`/api/knowledge-graph?schema_name=${schema}`);
      
      // Visualize the graph
      createForceGraph(response.data, svgRef.current);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const createForceGraph = (data, svgElement) => {
    // Clear previous visualization
    d3.select(svgElement).selectAll('*').remove();
    
    const width = 800;
    const height = 600;
    
    // Create the SVG container
    const svg = d3.select(svgElement)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);
    
    // Create the force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40));
    
    // Create the links
    const link = svg.append('g')
      .selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => d.type === 'direct' ? 2 : 1)
      .attr('stroke-dasharray', d => d.type === 'inferred' ? '5,5' : null);
    
    // Create the nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(data.nodes)
      .join('g')
      .call(drag(simulation));
    
    // Add circles to nodes
    node.append('circle')
      .attr('r', 15)
      .attr('fill', d => d.type === 'table' ? '#4299e1' : '#ed8936');
    
    // Add labels to nodes
    node.append('text')
      .attr('dx', 18)
      .attr('dy', '.35em')
      .text(d => d.name)
      .attr('font-size', '10px');
    
    // Add titles for tooltips
    node.append('title')
      .text(d => `${d.name}: ${d.description || 'No description'}`);
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    
    // Drag behavior function
    function drag(simulation) {
      function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }
      
      function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }
      
      function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }
      
      return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>Schema Visualizer</title>
        <meta name="description" content="Visualize SAP HANA schema as a knowledge graph" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-center mb-8">
          Schema Visualizer
        </h1>
        
        <div className="bg-white p-6 rounded-lg shadow-md max-w-5xl mx-auto">
          <div className="mb-4">
            <label htmlFor="schema" className="block text-gray-700 mb-2">
              Schema Name
            </label>
            <div className="flex">
              <input
                type="text"
                id="schema"
                value={schema}
                onChange={(e) => setSchema(e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter SAP HANA schema name"
              />
              <button
                onClick={handleVisualize}
                disabled={loading}
                className="px-4 py-2 bg-blue-500 text-white rounded-r-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              >
                {loading ? 'Loading...' : 'Visualize'}
              </button>
            </div>
          </div>
          
          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
              {error}
            </div>
          )}
          
          <div className="mt-6 border border-gray-200 rounded-md overflow-hidden">
            <svg ref={svgRef} className="w-full h-[600px] bg-white"></svg>
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            <p className="mb-2">Visualization Legend:</p>
            <ul className="list-disc pl-5">
              <li><span className="inline-block w-3 h-3 rounded-full bg-blue-500 mr-2"></span> Table</li>
              <li><span className="inline-block w-3 h-3 rounded-full bg-orange-500 mr-2"></span> View</li>
              <li><span className="inline-block w-4 h-px bg-gray-500 mr-2"></span> Direct Relationship</li>
              <li><span className="inline-block w-4 h-px bg-gray-500 mr-2 border-dashed border-t-2"></span> Inferred Relationship</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
EOF
    
    # Create a simple API test endpoint
    cat > pages/api/health.js << 'EOF'
export default function handler(req, res) {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
}
EOF
    
    # Create Vercel configuration
    cat > vercel.json << 'EOF'
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
      "dest": "https://BACKEND_URL_PLACEHOLDER/api/v1/$1"
    }
  ],
  "env": {
    "BACKEND_URL": "https://BACKEND_URL_PLACEHOLDER"
  }
}
EOF
    
    # Return to the original directory
    cd ..
else
    echo -e "${YELLOW}Using existing frontend directory at ${FRONTEND_DIR}.${NC}"
fi

# Create deployment script
echo -e "${YELLOW}Step 3: Creating deployment script...${NC}"

cat > deploy_frontend.sh << 'EOF'
#!/bin/bash
# Deploy the Next.js frontend to Vercel

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if backend URL is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Backend URL is required${NC}"
    echo "Usage: ./deploy_frontend.sh <backend_url>"
    exit 1
fi

BACKEND_URL=$1

# Navigate to frontend directory
cd owl-frontend

# Update .env.production with backend URL
echo -e "${YELLOW}Updating backend URL in .env.production...${NC}"
sed -i "" "s|BACKEND_URL=.*|BACKEND_URL=${BACKEND_URL}|g" .env.production
sed -i "" "s|NEXT_PUBLIC_BACKEND_URL=.*|NEXT_PUBLIC_BACKEND_URL=${BACKEND_URL}|g" .env.production

# Update vercel.json with backend URL
echo -e "${YELLOW}Updating backend URL in vercel.json...${NC}"
sed -i "" "s|BACKEND_URL_PLACEHOLDER|${BACKEND_URL}|g" vercel.json

# Deploy to Vercel
echo -e "${YELLOW}Deploying to Vercel...${NC}"
vercel --prod

echo -e "${GREEN}Frontend deployment complete!${NC}"
exit 0
EOF

chmod +x deploy_frontend.sh

# Create local development script
cat > start_frontend_dev.sh << 'EOF'
#!/bin/bash
# Start the Next.js frontend in development mode

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if backend URL is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}No backend URL provided. Using default (http://localhost:8000).${NC}"
    BACKEND_URL="http://localhost:8000"
else
    BACKEND_URL=$1
fi

# Navigate to frontend directory
cd owl-frontend

# Update .env.local with backend URL
echo -e "${YELLOW}Updating backend URL in .env.local...${NC}"
sed -i "" "s|BACKEND_URL=.*|BACKEND_URL=${BACKEND_URL}|g" .env.local
sed -i "" "s|NEXT_PUBLIC_BACKEND_URL=.*|NEXT_PUBLIC_BACKEND_URL=${BACKEND_URL}|g" .env.local

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Start development server
echo -e "${GREEN}Starting development server...${NC}"
echo -e "Frontend will be available at http://localhost:3000"
npm run dev
EOF

chmod +x start_frontend_dev.sh

echo -e "${GREEN}Stage 3 (Vercel Frontend Setup) complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Start the NVIDIA backend with ./start_nvidia_backend.sh"
echo "2. Start the frontend development server with ./start_frontend_dev.sh [backend_url]"
echo "3. To deploy to Vercel, run ./deploy_frontend.sh <backend_url>"

echo -e "\n${BLUE}To test the full system locally:${NC}"
echo -e "1. ./start_nvidia_backend.sh"
echo -e "2. ./start_frontend_dev.sh http://localhost:8000"
echo -e "3. Open http://localhost:3000 in your browser"

exit 0