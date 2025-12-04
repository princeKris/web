# Angular + Django PDF Social Platform

A full-stack web application where an Angular SPA frontend communicates with a Django REST API backend using JWT authentication. [file:1]  
The app provides a PDF-centric social platform with user profiles, PDF uploads with auto-generated previews, comments, likes, leaderboards, Google OAuth, and an infinite-scroll home feed. [file:1]

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [API Highlights](#api-highlights)
- [Getting Started](#getting-started)
  - [Backend Setup (Django)](#backend-setup-django)
  - [Frontend Setup (Angular)](#frontend-setup-angular)
- [Core Workflows](#core-workflows)
- [Security & Permissions](#security--permissions)
- [License](#license)

## Overview

This project combines an Angular single-page application with a Django REST API to build a modern PDF-sharing and discussion platform. [file:1]  
Angular handles the UI, routing, and auth state, while Django manages authentication, role-based permissions, file storage, and REST endpoints. [file:1]

## Features

- User registration and JWT login, plus optional Google OAuth 2.0 login. [file:1]  
- Custom user model with roles (admin, staff, user), profile image, bio, verification and block flags. [file:1]  
- Upload PDF posts with automatic first-page image thumbnail generation using PyMuPDF. [file:1]  
- Comment system with threaded replies and like/unlike functionality for posts. [file:1]  
- Infinite-scroll home feed, search for users and posts, and leaderboards for top users and top posts by likes. [file:1]  
- Unified role-based permission system for users, posts, and comments. [file:1]

## Tech Stack

- **Frontend:** Angular SPA (components, services, guards, HTTP interceptor for JWT). [file:1]  
- **Backend:** Django, Django REST Framework, SimpleJWT. [file:1]  
- **Auth:** JWT access/refresh tokens, Google OAuth 2.0. [file:1]  
- **Media & Files:** PyMuPDF (fitz), optional pdf2image, Pillow for image handling. [file:1]  
- **Database:** SQLite by default, configurable to other databases via Django settings. [file:1]

## Architecture

### Angular Application

- **Components:**  
  - `Nav`: authentication (login/signup/Google login), search, profile menu, navigation. [file:1]  
  - `Home`: homepage feed with pagination/infinite scroll, top users, and top posts. [file:1]  
  - `UserProfile`: view and edit profile, upload profile image, create and delete own PDF posts. [file:1]  
  - `ViewPdfPost`: single PDF view with preview, comments, likes, and download counter. [file:1]  
  - `Footer`: static footer component. [file:1]

- **Services:**  
  - `AuthService`: login, register, Google login, token storage, and user observable. [file:1]  
  - `ProfileService`: profile, posts, comments, likes, search, leaderboards, and home feed operations. [file:1]

- **Guards & Interceptors:**  
  - `LoginGuard`: protects authenticated-only routes. [file:1]  
  - JWT interceptor: injects `Authorization: Bearer <token>` into HTTP requests. [file:1]

### Django API

- **Models:**  
  - `User`: custom `AbstractUser` with role, image, bio, verified/blocked flags and timestamps. [file:1]  
  - `Post`: PDF file, generated preview image, author, likes, timestamps. [file:1]  
  - `Comment`: linked to post and user, optional parent for threaded replies. [file:1]

- **Serializers:**  
  - Auth: custom token serializer adding id, username, email, role, image to JWT login response; registration serializer for user creation. [file:1]  
  - Profile: serializers for reading/updating profile with absolute media URLs. [file:1]  
  - Posts/Comments: post, home-feed, and mini comment serializers, plus mini/leaderboard/search serializers. [file:1]

- **Permissions:**  
  - `UnifiedRBACPermission`: single RBAC layer for all views, with public read, authenticated write, owner checks, admin/staff override. [file:1]

## API Highlights

Main endpoints (high level): [file:1]

- **Auth:**  
  - `POST auth/login` – JWT login.  
  - `POST auth/register` – user registration + tokens.  
  - `POST auth/google` – Google OAuth exchange.

- **Profile:**  
  - `GET me` – authenticated user profile.  
  - `PATCH profile/update` – update profile (bio, image, etc.).  
  - `GET profile/<id>` – public profile with posts.  
  - `PATCH profile/verify/<id>` – admin verify/unverify user.

- **Posts & Interactions:**  
  - `GET/POST posts` – list/create user’s posts.  
  - `PATCH/DELETE posts/<id>` – update/delete post.  
  - `GET posts/single/<id>` – public single-post view.  
  - `POST posts/<id>/like` – toggle like.  
  - `POST posts/<id>/comment` – add comment.  
  - `DELETE posts/<post_id>/comment/<comment_id>` – delete comment.

- **Discovery:**  
  - `GET search?q=` – search users and posts.  
  - `GET topusers` – top users by total likes.  
  - `GET topposts` – most liked posts.  
  - `GET home?page=&limit=` – paginated home feed.

## Getting Started

### Backend Setup (Django)

1. Create and activate a Python virtual environment.  
2. Install backend dependencies (Django, djangorestframework, django-cors-headers, djangorestframework-simplejwt, PyMuPDF, Pillow, pdf2image, Google auth libraries). [file:1]  
3. Configure environment variables and `settings.py` (SECRET_KEY, database, CORS, media paths, `GOOGLE_CLIENT_ID`). [file:1]  
4. Run migrations and create a superuser. [file:1]  
5. Start the development server with `python manage.py runserver`. [file:1]

### Frontend Setup (Angular)

1. Install Node.js dependencies (`npm install` or equivalent).  
2. Configure Angular environment files with the Django API base URL and Google client ID. [file:1]  
3. Run the Angular dev server (for example, `ng serve`).  
4. Open the app in the browser and ensure CORS is configured correctly on the backend. [file:1]

## Core Workflows

- **Authentication:**  
  - Users register or log in to obtain JWT access/refresh tokens and user metadata (id, username, email, role, image). [file:1]  
  - Google login verifies an ID token and returns JWT tokens plus profile info. [file:1]

- **PDF Posts & Previews:**  
  - Authenticated users upload PDFs through the posts endpoint. [file:1]  
  - After saving, the backend generates a first-page preview image using PyMuPDF and stores it alongside the PDF. [file:1]

- **Social Features:**  
  - Users like/unlike posts, add/delete comments, and explore trending posts and top creators via leaderboard endpoints. [file:1]

## Security & Permissions

- All write operations and protected profile endpoints require JWT authentication. [file:1]  
- `UnifiedRBACPermission` enforces:  
  - Public read access for safe methods.  
  - Authenticated-only create/update/delete.  
  - Owners can manage their own posts, comments, and profiles, while admins/staff can moderate more broadly. [file:1]

## License

This project respects all relevant third-party licenses and intellectual property. [web:6][file:1]  
Review the licenses of Angular, Django, DRF, SimpleJWT, PyMuPDF, Pillow, pdf2image, and Google auth libraries before using this project in production. [web:6][file:1]
