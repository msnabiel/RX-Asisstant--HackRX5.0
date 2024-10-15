# PS4-GPTeam-RX-Asisstant

This repository contains a powerful assistant built using Flask and various machine learning models to extract key information from user queries and documents. It supports various document formats such as PDF, PPT, DOCX, and images, allowing users to upload and extract relevant information dynamically.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)
- [Author Information](#author-information)

## Demonstration
[Watch the Video](https://github.com/msnabiel/RX-Asisstant/raw/main/References/recording.mov)

## Features
- Extracts key information from various document types using Optical Character Recognition (OCR).
- Supports multiple actions such as creating orders, checking eligibility, generating leads, and more.
- Implements a conversational interface that keeps track of user sessions and pending actions.
- Uses the Gemini and Flan-T5 models for understanding and processing queries.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/msnabiel/RX-Asisstant.git
   cd PS4-GPTeam
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Google API key:**
   ```bash
   export GOOGLE_API_KEY='YOUR_API_KEY'
   ```

## Usage
To run the application, follow these steps:

1. Start the Flask application:
   ```bash
   python backend.py
   ```

   The application will start on `http://localhost:3000`.

2. Run the Chainlit interface:
   ```bash
   chainlit run frontend.py
   ```

   You can now chat with the assistant through the Chainlit interface.

## API Endpoints

### Upload Document
- **POST** `/upload_document`
- **Description**: Uploads a document and extracts text.
- **Parameters**: 
  - `document`: The document file to upload (supports PDF, PPT, DOCX, images, etc.).

### Chat
- **POST** `/chat`
- **Description**: Sends a query to the assistant and receives a response.
- **Headers**:
  - `x-user-id`: Unique identifier for the user.
  - `x-session-id`: Unique identifier for the user session.
- **Parameters**: 
  - `query`: The user's query as a string.
  - `document_id`: Optional document ID to use for context.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. Contributions are welcome!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author Information
- **Author**: [msnabiel](https://github.com/msnabiel)
- **Email**: [msyednabiel@gmail.com](mailto:msyednabiel@gmail.com)

