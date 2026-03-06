<h1>OrientAR Chatbot</h1>

<h2>Project Overview</h2>
<p>
  OrientAR Chatbot is an AI-powered assistant developed for the OrientAR mobile application at
  METU Northern Cyprus Campus.
</p>
<p>
  The chatbot helps students quickly access campus-related information such as announcements,
  events, facilities, and general university services. It uses a Retrieval-Augmented Generation
  (RAG) architecture to provide grounded responses based only on the available campus data.
</p>
<p>
  The system retrieves relevant information from a knowledge base stored in Firebase Firestore,
  converts it into embeddings, and generates responses using a Large Language Model (LLM)
  running through Ollama.
</p>

<h2>Features</h2>
<ul>
  <li>AI-powered campus information assistant</li>
  <li>Retrieval-Augmented Generation (RAG) architecture</li>
  <li>Firebase Firestore knowledge base</li>
  <li>Local LLM inference using Ollama</li>
  <li>Vector similarity search using Chroma</li>
  <li>RESTful API for mobile application integration</li>
</ul>

<h2>Technologies Used</h2>
<ul>
  <li>Python</li>
  <li>FastAPI</li>
  <li>Ollama</li>
  <li>Chroma Vector Database</li>
  <li>Firebase Firestore</li>
  <li>Retrieval-Augmented Generation (RAG)</li>
</ul>

<h2>System Architecture</h2>

<h3>Backend API</h3>
<p>
  The backend is implemented using FastAPI and provides RESTful endpoints for processing user
  queries from the mobile application.
</p>

<h3>Knowledge Base</h3>
<p>
  Campus-related information is stored in Firebase Firestore. This data is retrieved and used
  as context for answering user questions.
</p>

<h3>LLM and Embeddings</h3>
<p>
  The system uses Ollama to run the Large Language Model and generate embeddings. The embeddings
  are stored in Chroma Vector Database, which allows efficient similarity search during query
  processing.
</p>

<h2>Requirements</h2>
<ul>
  <li>Python 3.11+</li>
  <li>pip</li>
  <li>Ollama runtime</li>
  <li>Internet connection for Firebase Firestore access</li>
</ul>

<h2>Environment Configuration</h2>
<p>
  For security reasons, Firebase credentials are not stored in the repository.
</p>
<p>
  Before running the server, configure the following environment variables:
</p>

<pre><code>FIREBASE_SA_B64
LLM_BASE_URL
LLM_MODEL
EMBEDDING_MODEL
EMBEDDING_BASE_URL
CHROMA_DIR</code></pre>

<p>
  The Firebase Service Account JSON file must be encoded in Base64 format and assigned to the
  <code>FIREBASE_SA_B64</code> environment variable.
</p>

<h2>Installation</h2>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>Running the Server</h2>
<p>
  Make sure that Ollama is running and the required model is available.
</p>

<pre><code>uvicorn main:app --host 0.0.0.0 --port 8000</code></pre>

<p>
  After the server starts successfully, the REST API endpoints become available for handling
  chatbot queries.
</p>
<p>
  The OrientAR mobile application communicates with this backend to send user questions and
  receive responses in real time.
</p>

<h2>Future Improvements</h2>
<ul>
  <li>Faster response generation using GPU-based models</li>
  <li>Expansion of the knowledge base</li>
  <li>Improved retrieval accuracy</li>
  <li>Deployment on scalable cloud infrastructure</li>
</ul>
