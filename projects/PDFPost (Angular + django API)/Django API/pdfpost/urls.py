from django.urls import path
from .views import (LoginView, RegisterView, MeView, UpdateProfile,ListCreatePost, DetailPost, ToggleLike, GetProfileById, SinglePostView, AddCommentView,
                    SearchView,TopUsersByLikes,TopPostsAPIView, PaginatedHomePosts,ToggleVerifyView,GoogleAuthAPIView)
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path("google-login/", GoogleAuthAPIView.as_view()),
    path("login/", LoginView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view()),
    path("register/", RegisterView.as_view(), name="register"),
    path('me/', MeView.as_view(), name='me'),
    path("update-profile/", UpdateProfile.as_view(), name="update-profile"),
    path("mypost/", ListCreatePost.as_view(), name="my-post"),
    path("myposts/<int:pk>/", DetailPost.as_view(), name="my-posts"),
   # path("homeposts/", ListHomePost.as_view(), name="home-posts"),
    path("getprofile/<int:pk>/", GetProfileById.as_view()),
    path("post/<int:pk>/", SinglePostView.as_view()),
    path("post/<int:pk>/comment/", AddCommentView.as_view()),
    path("post/<int:post_id>/comment/<int:comment_id>/", AddCommentView.as_view()),
    path("posts/<int:post_id>/like/", ToggleLike.as_view()),
    path("search/", SearchView.as_view()),
    path("top-users-by-likes/", TopUsersByLikes.as_view(), name="top-users-by-likes"),
    path('top-posts/', TopPostsAPIView.as_view()),
    path("users/<int:pk>/toggle-verify/", ToggleVerifyView.as_view()),
    path("home-posts/", PaginatedHomePosts.as_view()),  

]
