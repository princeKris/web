from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.conf import settings
from .models import Post,Comment
User = get_user_model()


class MyTokenObtainPairSerializer(TokenObtainPairSerializer):

    def validate(self, attrs):
        data = super().validate(attrs)
        data["id"] = self.user.id
        data["username"] = self.user.username
        data["email"] = self.user.email
        data["role"] = self.user.role
        request = self.context.get("request", None)
        if self.user.image:
            if request:
                data["image"] = request.build_absolute_uri(self.user.image.url)
            else:
                data["image"] = self.user.image.url
        else:
            data["image"] = None    
        return data


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=4)
    
    class Meta:
        model = User
        fields = ["id", "username", "email", "password"]
    
    def create(self, validated_data):
        user = User(
            username=validated_data["username"],
            email=validated_data["email"]
        )
        user.set_password(validated_data["password"])
        user.save()
        return user
    

class MeSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "bdo",
            "bio",
            "image",
            "verified",
            "role",
        ]
        read_only_fields = ['id','verified']
        
        def to_representation(self, instance):
            data = super().to_representation(instance)
            request = self.context.get("request")
            if instance.image:
                data['image'] = request.build_absolute_uri(instance.image.url)
            else:
                data['image'] = None
            return data


class UserUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id","username", "email", "bio", "bdo", "image"]
        read_only_fields = ['id']
        extra_kwargs = {
            "email": {"required": False},
            "username": {"required": False},
            "bio": {"required": False},
            "bdo": {"required": False},
            "image": {"required": False},
        }
    
    def validate_email(self, value):
        user = self.context['request'].user
        if User.objects.exclude(id=user.id).filter(email=value).exists():
            raise serializers.ValidationError("Email already exists.")
        return value
    

class PostSerializer(serializers.ModelSerializer):
    file = serializers.SerializerMethodField()
    image = serializers.SerializerMethodField()
    is_liked=serializers.SerializerMethodField()
    likes_count=serializers.SerializerMethodField()
    comments = serializers.SerializerMethodField() 
    
    class Meta:
        model = Post
        fields = [
            "id",
            "title",
            "description",
            "file",
            "image",
            "author",
            "likes",
            "is_liked",
            "likes_count",
            "comments",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["author", "likes", "created_at", "updated_at"]
    
    def get_file(self, obj):
        if obj.file:
            return settings.MEDIA_URL + obj.file.name
        return None
    
    def get_is_liked(self, obj):
        request = self.context.get("request", None)
        if not request or not hasattr(request, "user"):
            return False
        user = request.user
        if user.is_authenticated:
            return obj.likes.filter(id=user.id).exists()
        return False
    
    # Count likes
    def get_likes_count(self, obj):
        return obj.likes.count()
    
    # GET COMMENTS
    def get_comments(self, obj):
        comments = obj.comments.filter(parent=None).order_by("-created_at")
        return CommentMiniSerializer(
            comments, many=True, context=self.context
        ).data

    def get_image(self, obj):
        if obj.image:
            return settings.MEDIA_URL + obj.image.name
        return None
    
    def to_representation(self, instance):
            data = super().to_representation(instance)
            request = self.context.get("request")
            if instance.image:
                data['image'] = request.build_absolute_uri(instance.image.url)
            else:
                data['image'] = None
            if instance.file:
                data['file'] = request.build_absolute_uri(instance.file.url)
            else:
                data['file'] = None
            return data


class CreatePostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ["title", "description", "file"]


class UserMiniSerializer(serializers.ModelSerializer):
    image = serializers.SerializerMethodField()
    class Meta:
        model = User
        fields = ["id", "username", "image","verified"]
    def get_image(self, obj):
        request = self.context.get("request")
        if obj.image:
            return request.build_absolute_uri(obj.image.url)
        return None
    

class CommentMiniSerializer(serializers.ModelSerializer):
    user = UserMiniSerializer(read_only=True)

    class Meta:
        model = Comment
        fields = [
            "id",
            "text",
            "created_at",
            "user",
        ]


class HomePostSerializer(serializers.ModelSerializer):
    author = UserMiniSerializer(read_only=True)
    file = serializers.SerializerMethodField()
    image = serializers.SerializerMethodField()
    likes_count = serializers.SerializerMethodField()
    comments = serializers.SerializerMethodField()
    is_liked = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            "id",
            "title",
            "description",
            "file",
            "image",
            "updated_at",
            "author",
            "likes_count",
            "is_liked",
            "comments",
        ]

    # Absolute URL for file
    def get_file(self, obj):
        request = self.context.get("request")
        if obj.file:
            return request.build_absolute_uri(obj.file.url)
        return None

    # Absolute URL for preview image
    def get_image(self, obj):
        request = self.context.get("request")
        if obj.image:
            return request.build_absolute_uri(obj.image.url)
        return None

    # Count likes
    def get_likes_count(self, obj):
        return obj.likes.count()
    
      # NEW: Check if logged-in user already liked the post
    def get_is_liked(self, obj):
        request = self.context.get("request", None)
        if not request or not hasattr(request, "user"):
            return False
        user = request.user
        if user.is_authenticated:
            return obj.likes.filter(id=user.id).exists()
        return False

    # Only top-level comments with user info
    def get_comments(self, obj):
        comments = obj.comments.filter(parent=None).order_by("-created_at")
        return CommentMiniSerializer(comments, many=True, context=self.context).data


class SearchUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username","image","verified"]


class   SearchPostSerializer(serializers.ModelSerializer):
    author = SearchUserSerializer(read_only=True)
    class Meta:
        model = Post
        fields = [
            "id",
            "title",
            "description",
            "author"
        ]


class TopUserSerializer(serializers.ModelSerializer):
    total_likes = serializers.IntegerField(read_only=True)  

    class Meta:
        model = User
        fields = ["id", "username", "total_likes","verified"]


class TopPostSerializer(serializers.ModelSerializer):
    author_id = serializers.IntegerField(source="author.id", read_only=True)
    author_username = serializers.CharField(source="author.username", read_only=True)
    likes_count = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            "id",
            "title",
            "author_id",
            "author_username",
            "likes_count",
        ]

    def get_likes_count(self, obj):
        return obj.likes.count()
