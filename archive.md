---
layout: page
---

<h3 style="color:SlateBlue;">[Old] Blog posts</h3>

{% for post in site.categories.Unizar %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ site.baseurl }}{{ post.url }})
{% endfor %}
