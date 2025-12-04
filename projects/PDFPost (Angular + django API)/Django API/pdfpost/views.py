from django.shortcuts import render
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import (MyTokenObtainPairSerializer, RegisterSerializer, MeSerializer, UserUpdateSerializer, PostSerializer, CreatePostSerializer, HomePostSerializer, CommentMiniSerializer,
                          SearchPostSerializer,SearchUserSerializer, TopUserSerializer,TopPostSerializer)
from rest_framework.views import APIView
from django.db.models import Q
from rest_framework.response import Response
from rest_framework import status
from .models import Post,Comment
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from django.db.models import Count
from .permissions import UnifiedRBACPermission
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from django.conf import settings
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model


User = get_user_model()


class GoogleAuthAPIView(APIView):
    permission_classes = []  # Allow public access

    def post(self, request):
        token = request.data.get("id_token")

        if not token:
            return Response({"error": "Missing id_token"}, status=400)

        try:
            # Verify token
            google_info = id_token.verify_oauth2_token(
                token,
                grequests.Request(),
                settings.GOOGLE_CLIENT_ID
            )
        except Exception:
            return Response({"error": "Invalid Google token"}, status=400)

        email = google_info.get("email")
        name = google_info.get("name", "")
        picture = google_info.get("picture")

        if not email:
            return Response({"error": "Google email not found"}, status=400)

        # Find existing user or create new one
        user = User.objects.filter(email=email).first()
        created = False

        if not user:
            # Create new user (NO password)
            base_username = email.split("@")[0]
            username = base_username
            i = 1

            # Ensure unique username
            while User.objects.filter(username=username).exists():
                username = f"{base_username}{i}"
                i += 1

            user = User(username=username, email=email)
            user.set_unusable_password()
            user.save()
            created = True

        # Optional: Update name only if user has no name set
        if name and not user.first_name:
            user.first_name = name
            user.save()

        if user.image:
            image_url = request.build_absolute_uri(user.image.url)
        else:
            image_url =None

        # Issue JWT tokens
        refresh = RefreshToken.for_user(user)
        return Response({
            "created": created,
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "image": image_url,  
            "access": str(refresh.access_token),
            "refresh": str(refresh),
        }, status=200)
    
    
class LoginView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


class RegisterView(APIView):

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)

        if serializer.is_valid():
            user = serializer.save()
            # Generate JWT tokens for the new user
            from rest_framework_simplejwt.tokens import RefreshToken
            refresh = RefreshToken.for_user(user)
            return Response({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "message": "User registered successfully"
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MeView(APIView):
    permission_classes = [UnifiedRBACPermission]

    def get(self, request):
        serializer = MeSerializer(request.user, context={'request': request})
        return Response(serializer.data)


class UpdateProfile(APIView):
    permission_classes = [UnifiedRBACPermission]
    parser_classes = [MultiPartParser, FormParser]

    def patch(self, request):
        user = request.user
        serializer = UserUpdateSerializer(user, data=request.data, partial=True,context={"request": request})

        if serializer.is_valid():
            serializer.save()
            return Response({"message": "Profile updated successfully", "user": serializer.data})

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

class ListCreatePost(APIView):
    permission_classes = [UnifiedRBACPermission]

    def get(self, request):
        # Return only logged-in user's posts
        posts = Post.objects.filter(author=request.user).order_by("-created_at")
        serializer = PostSerializer(posts, many=True, context={"request": request})
        return Response(serializer.data)

    def post(self, request):
        serializer = CreatePostSerializer(data=request.data)

        if serializer.is_valid():
            post = serializer.save(author=request.user)
            post.save()  # triggers PDF preview image generation

            return Response(PostSerializer(post,context={"request": request}).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    
class DetailPost(APIView):
    permission_classes = [UnifiedRBACPermission] 
    def patch(self, request,pk):
        post = get_object_or_404(Post, pk=pk)
        # Enforce RBAC
        self.check_object_permissions(request, post)
        serializer = CreatePostSerializer(post, data=request.data, partial=True)
        if serializer.is_valid():
            updated = serializer.save()
            return Response(PostSerializer(updated, context={"request": request}).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        post = get_object_or_404(Post, pk=pk)
        # Enforce RBAC
        self.check_object_permissions(request, post)
        post.delete()
        return Response({"message": "Post deleted"})


class ToggleLike(APIView):
    permission_classes = [UnifiedRBACPermission]

    def post(self, request, post_id):
        try:
            post = Post.objects.get(id=post_id)
        except Post.DoesNotExist:
            return Response({"error": "Not found"}, status=404)

        if request.user in post.likes.all():
            post.likes.remove(request.user)
            liked = False
        else:
            post.likes.add(request.user)
            liked = True

        return Response({"liked": liked, "likes_count": post.likes.count()})
    

class GetProfileById(APIView):
    permission_classes = [UnifiedRBACPermission]
    def get(self, request, pk):
        User = get_user_model()
        # Get user
        user = get_object_or_404(User, id=pk)
        # Get all posts by that user
        posts = Post.objects.filter(author=user).order_by("-created_at")
        # Serialize posts using your existing HomePostSerializer
        post_serializer = HomePostSerializer(
            posts,
            many=True,
            context={"request": request}
        )
        # Return combined response
        return Response({
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "verified":user.verified,
                "bio": getattr(user, "bio", ""),
                "image": request.build_absolute_uri(user.image.url) if user.image else None,
            },
            "posts": post_serializer.data,
        })
    
    def delete(self, request, pk):
        User = get_user_model()
        user_to_delete = get_object_or_404(User, id=pk)
        # Check RBAC rules
        self.check_object_permissions(request, user_to_delete)
        user_to_delete.delete()
        return Response({"message": "User deleted successfully"}, status=200)
    

class SinglePostView(APIView):
    #permission_classes = [AllowAny]  # Public post page
    permission_classes = [UnifiedRBACPermission]
    def get(self, request, pk):
        try:
            post = Post.objects.get(pk=pk)
        except Post.DoesNotExist:
            return Response({"error": "Post not found"}, status=404)

        serializer = HomePostSerializer(post, context={"request": request})
        return Response(serializer.data)


class AddCommentView(APIView):
    permission_classes = [UnifiedRBACPermission]

    def post(self, request, pk):
        text = request.data.get("text")

        if not text:
            return Response({"error": "Text is required"}, status=400)
        
        post = get_object_or_404(Post, pk=pk)
        comment = Comment.objects.create(
            user=request.user,
            post=post,
            text=text
        )
        return Response(
            CommentMiniSerializer(comment, context={"request": request}).data,
            status=201
        )
    
    def delete(self, request, post_id, comment_id):
        comment = get_object_or_404(Comment, id=comment_id, post_id=post_id)
        # Apply RBAC — ALLOWS admin/staff rules correctly
        self.check_object_permissions(request, comment)
        comment.delete()
        return Response({"message": "Comment deleted"}, status=200)


User = get_user_model()
class SearchView(APIView):

    def get(self, request):
        query = request.GET.get("q", "").strip()

        if query == "":
            return Response({"users": [], "posts": []})

        # Search users
        users = User.objects.filter(username__icontains=query)

        # Search posts
        posts = Post.objects.filter(
            Q(title__icontains=query) |
            Q(description__icontains=query)
        )

        return Response({
            "users": SearchUserSerializer(users, many=True, context={"request": request}).data,
            "posts": SearchPostSerializer(posts, many=True, context={"request": request}).data,
        })


class TopUsersByLikes(APIView):
    """
    Return top 5 users ordered by total likes across all posts they authored.
    Response: [{id, username, total_likes}, ...] (max 5)
    """

    def get(self, request):
        # Annotate each user with total likes across their posts:
        # `posts__likes` follows: User -> posts (related_name) -> likes (M2M)
        users_qs = (
            User.objects
                .annotate(total_likes=Count('posts__likes'))
                .order_by('-total_likes', 'username')[:5]
        )
        serializer = TopUserSerializer(users_qs, many=True, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class TopPostsAPIView(APIView):

    def get(self, request):
        posts = Post.objects.annotate(
            like_total=Count("likes")
        ).order_by("-like_total")[:5]

        serializer = TopPostSerializer(posts, many=True)
        return Response(serializer.data)


class PaginatedHomePosts(APIView):
    def get(self, request):
        page = int(request.GET.get("page", 1))
        limit = int(request.GET.get("limit", 10))
        start = (page - 1) * limit
        end = start + limit
        posts = Post.objects.all().order_by("-created_at")[start:end]
        total = Post.objects.count()
        serializer = HomePostSerializer(posts, many=True, context={"request": request})
        return Response({
            "page": page,
            "has_more": end < total,
            "results": serializer.data
        })

# class ListHomePost(APIView):
#     def get(self, request):
#         posts = Post.objects.all().order_by("-created_at")
#         serializer = HomePostSerializer(posts, many=True, context={"request": request})
#         return Response(serializer.data)


class ToggleVerifyView(APIView):
    permission_classes = [UnifiedRBACPermission]

    def patch(self, request, pk):
        User = get_user_model()
        try:
            user = User.objects.get(id=pk)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)

        if request.user.role != "admin":
            return Response({"error": "Only admin can verify/unverify users"}, status=403)

        user.verified = not user.verified
        user.save()

        return Response({
            "message": "User verified" if user.verified else "User unverified",
            "verified": user.verified
        })
