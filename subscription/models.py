from django.db import models
from django.utils import timezone
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.conf import settings
import stripe

# Configure the Stripe API key
stripe.api_key = settings.STRIPE_SECRET_KEY

class SubscriptionPlan(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    stripe_price_id = models.CharField(max_length=50, null=True, blank=True)
    duration_days = models.IntegerField(default=30)
    trial_days = models.IntegerField(default=0)
    free = models.BooleanField(default=False)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name

    def create_stripe_price(self):
        """
        Create or update a Stripe Price for this plan.
        """
        if self.free:
            return

        if self.stripe_price_id:
            try:
                stripe.Price.retrieve(self.stripe_price_id)
                return
            except stripe.error.InvalidRequestError:
                self.stripe_price_id = None

        try:
            product = stripe.Product.create(
                name=self.name,
                description=self.description,
            )

            price = stripe.Price.create(
                product=product.id,
                unit_amount=int(self.price * 100),
                currency="usd",
                recurring={"interval": "month"},
            )

            self.stripe_price_id = price.id
            self.save()
        except stripe.error.StripeError as e:
            print(f"Error creating Stripe price: {e.user_message}")

@receiver(post_save, sender=SubscriptionPlan)
def create_or_update_stripe_price(sender, instance, **kwargs):
    instance.create_stripe_price()


class Subscription(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='subscriptions')
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.SET_NULL, null=True)
    start_date = models.DateTimeField(default=timezone.now)
    end_date = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=(
        ('active', 'Active'),
        ('pending', 'Pending'),
        ('canceled', 'Canceled'),
        ('expired', 'Expired'),
    ), default='active')
    stripe_subscription_id = models.CharField(max_length=100, null=True, blank=True)  # Store Stripe subscription ID

    # def save(self, *args, **kwargs):
    #     # Ensure only one active subscription per user
    #     if self.status == 'active' and Subscription.objects.filter(user=self.user, status='active').exclude(id=self.id).exists():
    #         raise ValueError("User already has an active subscription.")
    #     super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.username} - {self.plan.name if self.plan else 'No Plan'}"



class Payment(models.Model):
    subscription = models.ForeignKey(Subscription, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=(
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ), default='pending')
    stripe_payment_intent_id = models.CharField(max_length=100, null=True, blank=True)  # Store Stripe payment intent ID

    def __str__(self):
        return f"{self.subscription.user.username} - {self.amount} - {self.status}"

